"""
scripts/train.py
----------------
Mission Control for SPECTRA Experiments.

Orchestrates the training pipeline using Hydra for configuration
management and PyTorch Lightning for scalable execution.

Patches applied (2026-02-27 NASA-tier hardening):
    - preflight_check(): validates LMDB paths, task configs, method names
      before any GPU compute is allocated (D6)
    - resume_from: CLI passthrough for checkpoint resumption (D7)
    - Smart ModelCheckpoint: monitors val/miou for NYUv2, val/total_loss
      for synthetic/clinical (D9)
"""

import os
import sys
from pathlib import Path

# --- Path Resolution ---
# Ensures the 'spectra' package is discoverable when run from project root.
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import hydra
import torch
import logging
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from spectra.data.datamodule import SPECTRADataModule
from spectra.engine.trainer import SPECTRAModule
from spectra.engine.callbacks import SpectralMonitoringCallback, GradientHealthCallback

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spectra.train")

from hydra.core.hydra_config import HydraConfig


# =============================================================================
# PRE-FLIGHT VALIDATION
# =============================================================================

def preflight_check(cfg: DictConfig, output_dir: Path) -> None:
    """
    Validate all training prerequisites BEFORE allocating GPU compute.

    Principle: Fail fast and informatively. A 10-second pre-flight check
    is infinitely cheaper than discovering a broken config at step 500.

    Raises:
        SystemExit with a descriptive message if any check fails.
    """
    errors = []

    # 1. NYUv2 LMDB must exist before training
    benchmark = cfg.get("dataset", {}).get("benchmark", "")
    if benchmark == "nyuv2":
        lmdb_train = Path(cfg.dataset.root) / "train" / "data.lmdb"
        lmdb_val   = Path(cfg.dataset.root) / "val"   / "data.lmdb"
        if not lmdb_train.exists():
            errors.append(
                f"NYUv2 train LMDB missing: {lmdb_train}\n"
                f"  → Run: python scripts/materialize_nyuv2.py"
            )
        if not lmdb_val.exists():
            errors.append(
                f"NYUv2 val LMDB missing: {lmdb_val}\n"
                f"  → Run: python scripts/materialize_nyuv2.py"
            )

    # 2. Tasks must be defined
    tasks = list(cfg.get("tasks", []))
    if not tasks:
        errors.append("cfg.tasks is empty — no tasks configured. Check your dataset config.")

    # 3. Method name must be valid
    valid_methods = {"bpgs", "bpgs_alb", "kendall", "uwso", "pcgrad", "ntkmtl", "static"}
    method_name = cfg.get("method", {}).get("name", "unknown")
    if method_name not in valid_methods:
        errors.append(
            f"Unknown method: '{method_name}'. "
            f"Valid: {sorted(valid_methods)}"
        )

    # 4. Loss names must be registered
    valid_losses = {"mse", "l1", "bce", "cross_entropy", "cosine",
                    "masked_l1", "cosine_dense"}
    for task_cfg in tasks:
        loss_name = task_cfg.get("loss", "mse")
        if loss_name not in valid_losses:
            errors.append(
                f"Task '{task_cfg.get('name', '?')}': unknown loss '{loss_name}'. "
                f"Valid: {sorted(valid_losses)}"
            )

    # 5. Resume checkpoint exists if specified
    ckpt_path = cfg.get("resume_from", None)
    if ckpt_path and not Path(ckpt_path).exists():
        errors.append(
            f"Resume checkpoint not found: {ckpt_path}\n"
            f"  → Check the path or remove 'resume_from' from config."
        )

    # 6. GPU memory sanity (warn only, do not block)
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size  = cfg.train.get("batch_size", 8)
        if benchmark == "nyuv2" and batch_size > 8 and free_mem_gb < 16.0:
            logger.warning(
                f"[PreFlight] batch_size={batch_size} on {free_mem_gb:.1f}GB GPU. "
                f"NYUv2/SegNet may OOM. Consider batch_size<=8 or "
                f"train.accumulate_grad_batches=2."
            )
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logger.info(f"[PreFlight] Multi-GPU detected: {num_gpus} GPUs. Using DDP strategy.")

    # --- Report and Exit on Failures ---
    if errors:
        logger.error("[PreFlight] FAILED with the following errors:")
        for i, e in enumerate(errors, 1):
            logger.error(f"  [{i}] {e}")
        raise SystemExit(
            f"\n\nPre-flight check FAILED ({len(errors)} error(s)). "
            f"Fix all errors above before training."
        )

    logger.info("[PreFlight] All systems nominal. GO for training. 🚀")


# =============================================================================
# CHECKPOINT FACTORY
# =============================================================================

def _dataset_key(cfg: DictConfig) -> str:
    """Canonical dataset identifier (backward-compatible)."""
    dcfg = cfg.get("dataset", {})
    return dcfg.get("name", dcfg.get("benchmark", "synthetic"))


def _merge_dataset_defaults(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """
    Merge dataset defaults with top-level overrides without struct-key crashes.

    OmegaConf structured nodes can reject unknown keys during direct merge
    (`ConfigKeyError`). We convert both to plain dict first, then recreate a
    DictConfig so CLI/top-level keys (e.g. `model.n_heads`) are preserved.
    """
    base = OmegaConf.to_container(base_cfg, resolve=False) if base_cfg is not None else {}
    override = OmegaConf.to_container(override_cfg, resolve=False) if override_cfg is not None else {}
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        override = {}
    # Deep merge on plain containers (not structured nodes) so nested sections
    # are preserved while top-level/CLI overrides still win.
    merged = OmegaConf.merge(base, override)
    return OmegaConf.create(merged)


def build_checkpoints(cfg: DictConfig, output_dir: Path):
    """
    Build ModelCheckpoint callbacks appropriate for the benchmark.

    Strategy:
        - NYUv2: Primary checkpoint by val/miou (the research metric);
                 also keep last.ckpt for resumption.
        - Synthetic/Clinical: Primary checkpoint by val/total_loss;
                              keep last.ckpt for resumption.

    Two separate checkpoints because we want BOTH the best-mIoU model
    (for reporting) and the last checkpoint (for resumption after crash).
    """
    ckpt_dir = output_dir / "checkpoints"
    benchmark = _dataset_key(cfg)

    checkpoints = []

    if benchmark == "nyuv2":
        # Best mIoU checkpoint — this is the model we report in the paper
        checkpoints.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-miou-ep{epoch:02d}-{val/miou:.4f}",
            monitor="val/miou",
            mode="max",
            save_top_k=3,
            save_last=False,
            auto_insert_metric_name=False,
        ))
        # Best loss checkpoint (secondary — useful when ALB ablations change metric)
        checkpoints.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-loss-ep{epoch:02d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ))
    elif benchmark == "clinical":
        # Clinical: select best model by outcome AUC (primary objective),
        # but still keep a loss-monitored 'last' checkpoint for robust resume.
        checkpoints.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-outcome-ep{epoch:02d}-{val/outcome_AUC:.4f}",
            monitor="val/outcome_AUC",
            mode="max",
            save_top_k=3,
            save_last=False,
            auto_insert_metric_name=False,
        ))
        checkpoints.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-loss-ep{epoch:02d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ))
    else:
        # Synthetic: loss is the primary signal
        checkpoints.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-ep{epoch:02d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ))

    return checkpoints


def build_early_stopping(cfg: DictConfig):
    """Optional early stopping; defaults on for clinical to prevent wasted epochs."""
    enabled = cfg.train.get("early_stop", None)
    benchmark = _dataset_key(cfg)
    if enabled is None:
        enabled = (benchmark == "clinical")
    if not enabled:
        return None

    if benchmark == "clinical":
        monitor, mode = "val/outcome_AUC", "max"
        min_delta = cfg.train.get("early_stop_min_delta", 1e-3)
    elif benchmark == "nyuv2":
        monitor, mode = "val/miou", "max"
        min_delta = cfg.train.get("early_stop_min_delta", 1e-4)
    else:
        monitor, mode = "val/total_loss", "min"
        min_delta = cfg.train.get("early_stop_min_delta", 1e-4)

    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=cfg.train.get("early_stop_patience", 3),
        min_delta=min_delta,
        check_on_train_epoch_end=False,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

class SOTAProgressBar(TQDMProgressBar):
    """
    Research-Grade Progress Bar (SOTA Style).
    Maps long metric keys to concise research shorthand (GN, L, AUC, etc.).
    """
    def init_train_tqdm(self) -> tqdm:
        bar = super().init_train_tqdm()
        # "Gold Standard" Format: Dense, no bars, high-fidelity telemetry
        bar.bar_format = "{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = super().init_validation_tqdm()
        bar.bar_format = "{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return bar

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None) # Remove v_num to save space
        
        # SOTA Shorthand Mapping
        mapping = {
            "train/total_loss": "L",
            "health/backbone_grad_norm": "GN",
            "pcgrad/total_conflicts": "C",
            "spectral/hf_divorce_index": "D",
            "health/backbone_weight_norm": "WN",
            "val/total_loss": "vL",
            "train/AUC": "AUC",
            "train/PRC": "PRC",
            "train/R": "R",
        }
        
        # Add task-specific adaptive mapping
        # NYUV2: mIoU, abs_rel, angle
        # Clinical: auc, prc, recall
        for task in getattr(pl_module, "task_names", []):
            mapping[f"train/{task}_loss"] = f"L_{task[:1]}"
            mapping[f"val/{task}_miou"] = "mIoU"
            mapping[f"val/{task}_abs_rel"] = "abs_rel"
            mapping[f"val/{task}_mean_angle"] = "angle"
            
        new_items = {}
        for k, v in items.items():
            # Handle PTL suffixes like _step or _epoch
            base_k = k.replace("_step", "").replace("_epoch", "")
            
            if base_k in mapping:
                new_items[mapping[base_k]] = v
            else:
                # Fallback: remove 'train/' or 'val/' prefix
                clean_k = k.replace("train/", "").replace("val/", "")
                new_items[clean_k] = v
        return new_items

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Environment & Seeding
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # 2. Workspace Preparation
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"[Mission-Control] Workspace: {output_dir}")
    logger.info(f"[Mission-Control] Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- NASA-Grade Config Merge ---
    # Hydra namespaces dataset configs under cfg.dataset.*.
    # 3. Pre-Flight Validation (D6)
    # Validates EVERYTHING before touching GPU. Fast fail saves compute.
    preflight_check(cfg, output_dir)

    # 4. Data Orchestration
    datamodule = SPECTRADataModule(cfg)

    # 5. Model Orchestration
    model = SPECTRAModule(cfg)

    # 6. Callback Infrastructure
    callbacks = [
        *build_checkpoints(cfg, output_dir),
        SpectralMonitoringCallback(log_every_n_epochs=5),
        GradientHealthCallback(check_interval=200),
        LearningRateMonitor(logging_interval="step"),
        SOTAProgressBar(refresh_rate=1),
    ]
    es_cb = build_early_stopping(cfg)
    if es_cb is not None:
        callbacks.append(es_cb)

    # 7. Logger Integration
    wandb_logger = None
    if cfg.get("logging", {}).get("use_wandb", False):
        # Deterministic Offline Mode (Silent Research)
        if cfg.logging.get("wandb_mode") == "offline":
            os.environ["WANDB_MODE"] = "offline"
            logger.info("[Logging] WandB Offline Mode Engaged (Silent Research)")

        wandb_logger = WandbLogger(
            project=cfg.logging.get("wandb_project", "spectra-mtl"),
            name=cfg.get("run_name", "unnamed_run"),
            save_dir=str(output_dir),
            offline=(cfg.logging.get("wandb_mode") == "offline"),
            log_model=False,
        )
        # Log the full config as a W&B artifact for exact reproducibility
        if wandb_logger.experiment is not None:
            wandb_logger.experiment.config.update(
                OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
            )

    # 8. Trainer Configuration
    # PCGrad uses manual optimization — gradient clipping must be done manually inside
    # the training_step. Setting gradient_clip_val here for non-PCGrad methods.
    gradient_clip_val = cfg.train.get("grad_clip", 1.0)
    if not getattr(model, "automatic_optimization", True):
        gradient_clip_val = None
        logger.info(
            "[Mission-Control] PCGrad detected (manual optimization). "
            "Automatic gradient clipping disabled — clipping handled inside training_step."
        )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp_find_unused_parameters_false" if torch.cuda.device_count() > 1 else "auto",
        precision=cfg.train.get("precision", "16-mixed"),
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.get("log_every_n_steps", 10),
        # Deterministic mode: correctness over speed during development.
        # Set deterministic=False in production for ~10-20% speed gain.
        deterministic=cfg.train.get("deterministic", False),
        # DDP: find_unused_parameters=False avoids the O(N) communication
        # overhead when all parameters are used in every forward pass.
        # If any parameters are conditionally unused, set to True.
    )

    # 9. Resume support (D7)
    # CLI: python scripts/train.py resume_from=path/to/last.ckpt
    ckpt_path = cfg.get("resume_from", None)
    if ckpt_path:
        logger.info(f"[Mission-Control] Resuming from checkpoint: {ckpt_path}")

    # 10. Mission Start
    logger.info(
        f"[Mission-Control] All systems GO. "
        f"Method={cfg.method.get('name', '?')}, "
        f"Epochs={cfg.train.epochs}, "
        f"Tasks={[t.name for t in cfg.tasks]}"
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("[Mission-Control] Mission Accomplished. ✓")


if __name__ == "__main__":
    main()
