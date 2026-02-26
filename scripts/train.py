"""
scripts/train.py
----------------
Mission Control for SPECTRA Experiments.

Orchestrates the training pipeline using Hydra for configuration 
management and PyTorch Lightning for scalable execution.
"""

import os
import sys
from pathlib import Path

# --- NASA-Grade Path Resolution ---
# Ensures the 'spectra' package is discoverable when run from the project root.
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import hydra
import torch
import logging
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from spectra.data.datamodule import SPECTRADataModule
from spectra.engine.trainer import SPECTRAModule
from spectra.engine.callbacks import SpectralMonitoringCallback, NTKGradExplosionTracker

# --- NASA-Grade Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spectra.train")

from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Environment & Seeding
    # Ensures absolute reproducibility across different hardware runs.
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    
    # --- NASA-Grade Config Merge ---
    # Hydra namespaces dataset configs. We explicitly bubble up overrides.
    if "model" in cfg.get("dataset", {}):
        cfg.model = OmegaConf.merge(cfg.model, cfg.dataset.model)
    if "tasks" in cfg.get("dataset", {}):
        cfg.tasks = cfg.dataset.tasks
    if "train" in cfg.get("dataset", {}):
        cfg.train = OmegaConf.merge(cfg.train, cfg.dataset.train)
    
    # 2. Workspace Preparation
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"[Mission-Control] Workspace Initialized: {output_dir}")
    logger.info(f"[Mission-Control] Config:\n{OmegaConf.to_yaml(cfg)}")

    # 3. Data Orchestration
    # SPECTRADataModule handles tiered acquisition (Cloud/Local/Build fallback).
    datamodule = SPECTRADataModule(cfg)
    
    # 4. Model Orchestration
    # SPECTRAModule integrates Backbone, ALB, Heads, and Weighter.
    model = SPECTRAModule(cfg)
    
    # We extract trainer-specific configs from Hydra to allow CLI overrides.
    trainer_node = cfg.get("trainer", OmegaConf.create({}))
    trainer_cfg = OmegaConf.to_container(trainer_node, resolve=True)
    
    # 5. Callback Infrastructure
    callbacks = [
        SpectralMonitoringCallback(log_every_n_epochs=1),
        NTKGradExplosionTracker()
    ]
    
    if trainer_cfg.get("logger", True) is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    if trainer_cfg.get("enable_checkpointing", True):
        callbacks.insert(0, ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="spectra-{epoch:02d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ))
    
    # 6. Logger Integration
    wandb_logger = None
    if cfg.get("logging", {}).get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=cfg.logging.get("wandb_project", "spectra-mtl"),
            name=cfg.get("run_name", "unnamed_run"),
            save_dir=output_dir,
            offline=cfg.logging.get("wandb_offline", False)
        )
    
    # 7. Trainer Orchestration
    # Using 'ddp' for multi-GPU efficiency, 'auto' for single-device fallbacks.
    trainer_node = cfg.get("trainer", OmegaConf.create({}))
    trainer_cfg = OmegaConf.to_container(trainer_node, resolve=True)
    
    # Automatic gradient clipping is not supported when manual optimization is used (e.g., PCGrad)
    gradient_clip_val = cfg.train.get("grad_clip", 1.0)
    if getattr(model, "automatic_optimization", True) is False:
        gradient_clip_val = None
        logger.info("[Mission-Control] Manual optimization detected. Disabling automatic gradient clipping.")

    # Base kwargs
    trainer_kwargs = dict(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        precision=cfg.train.get("precision", "16-mixed"),
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.get("log_every_n_steps", 10),
        deterministic=cfg.train.get("deterministic", False),
    )
    
    # Apply CLI overrides (e.g. trainer.fast_dev_run=True, trainer.accelerator=cpu)
    trainer_kwargs.update(trainer_cfg)
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # 8. Mission Start
    logger.info("[Mission-Control] All systems GO. Initiating training...")
    trainer.fit(model, datamodule=datamodule)
    logger.info("[Mission-Control] Mission Accomplished.")

if __name__ == "__main__":
    main()
