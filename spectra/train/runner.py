"""
spectra/train/runner.py
-----------------------
The Core Execution Runner.
Composes the pieces (Module, DataModule, Callbacks, Hydra) and executes.
"""

import os
import torch
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from hydra.utils import instantiate

from spectra.data.datamodule import SPECTRADataModule
from spectra.engine.callbacks import SpectralMonitoringCallback, GradientHealthCallback

from spectra.utils.callbacks import build_checkpoints, build_early_stopping
from spectra.utils.progress import SOTAProgressBar
from spectra.utils.config import _merge_dataset_defaults
from spectra.train.preflight import preflight_check

logger = logging.getLogger("spectra.runner")

def execute_training_mission(cfg: DictConfig, output_dir: Path):
    """
    The orchestrator for the entire SPECTRA training lifecycle.
    """
    # 1. Environment & Seeding
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    logger.info(f"[Mission-Control] Workspace: {output_dir}")
    logger.info(f"[Mission-Control] Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Configuration Integrity 
    # (Hydra @package _global_ now handles domain merging)

    # 3. Pre-Flight Validation
    preflight_check(cfg, output_dir)

    # 4. Data Orchestration
    datamodule = SPECTRADataModule(cfg)

    # 5. SOTA Hydra Target Injection
    # Instead of `model = SPECTRAModule(cfg)`, we let Hydra build the
    # specific Orthogonal engine AND the specific Domain Silo directly!
    # The `engine` is passed to the `module` as an injected dependency.
    engine = instantiate(cfg.module.engine)
    model = instantiate(cfg.module, cfg=cfg, engine=engine, _recursive_=False)

    # 6. Callback Infrastructure
    callbacks = [
        *build_checkpoints(cfg, output_dir),
        SpectralMonitoringCallback(log_every_n_epochs=5),
        GradientHealthCallback(check_interval=1),
        LearningRateMonitor(logging_interval="step"),
        SOTAProgressBar(refresh_rate=1),
    ]
    es_cb = build_early_stopping(cfg)
    if es_cb is not None:
        callbacks.append(es_cb)

    # 7. Logger Integration
    wandb_logger = None
    if cfg.get("logging", {}).get("use_wandb", False):
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
        if wandb_logger.experiment is not None:
            wandb_logger.experiment.config.update(
                OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
            )

    # 8. Trainer Configuration
    gradient_clip_val = cfg.train.get("grad_clip", 1.0)
    if not getattr(model, "automatic_optimization", True):
        gradient_clip_val = None # Managed by manual engine
        logger.info("[Mission-Control] PCGrad detected (manual optimization). Automatic PL clipping disabled.")

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
        deterministic=cfg.train.get("deterministic", False),
    )

    # 9. Mission Start
    ckpt_path = cfg.get("resume_from", None)
    if ckpt_path: logger.info(f"[Mission-Control] Resuming from checkpoint: {ckpt_path}")
    
    logger.info(
        f"[Mission-Control] All systems GO. "
        f"Method={cfg.get('method_name', cfg.get('method', {}).get('name', '?'))}, "
        f"Epochs={cfg.train.epochs}, "
        f"Tasks={[t.name for t in cfg.tasks]}"
    )
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("[Mission-Control] Mission Accomplished. [SUCCESS]")
