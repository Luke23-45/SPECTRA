"""
spectra/utils/callbacks.py
--------------------------
Universal Checkpoint and Early Stopping Factory.
"""

from pathlib import Path
from typing import List, Optional
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def _dataset_key(cfg: DictConfig) -> str:
    """Canonical dataset identifier (backward-compatible)."""
    dcfg = cfg.get("dataset", {})
    return dcfg.get("name", dcfg.get("benchmark", "synthetic"))

def build_checkpoints(cfg: DictConfig, output_dir: Path) -> List[ModelCheckpoint]:
    """
    Build ModelCheckpoint callbacks appropriate for the benchmark.
    """
    if not cfg.train.get("save_ckpt", True):
        return []

    ckpt_dir = output_dir / "checkpoints"
    benchmark = _dataset_key(cfg)
    checkpoints = []

    if benchmark == "nyuv2":
        checkpoints.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-miou-ep{epoch:02d}-{val/miou:.4f}",
            monitor="val/miou",
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
    elif benchmark == "clinical":
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

def build_early_stopping(cfg: DictConfig) -> Optional[EarlyStopping]:
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
