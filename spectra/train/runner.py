"""Training runner: composes Module, DataModule, Callbacks, and Trainer."""

import logging
import os
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from hydra.utils import instantiate

from spectra.data.datamodule import SPECTRADataModule
from spectra.engine.callbacks import GradientHealthCallback

from spectra.utils.callbacks import build_checkpoints, build_early_stopping
from spectra.utils.progress import build_progress_bar
from spectra.utils.config import _merge_dataset_defaults
from spectra.utils.seed import configure_reproducibility
from spectra.utils.wandb import WandbSession, WandbSettings
from spectra.train.artifacts import (
    resolve_artifact_dir,
    resolve_resume_checkpoint,
    stable_run_id,
    save_experiment_config,
    save_experiment_metadata,
    save_run_summary,
)
from spectra.train.preflight import preflight_check

logger = logging.getLogger("spectra.runner")

def execute_training_mission(cfg: DictConfig, output_dir: Path):
    """
    The orchestrator for the entire SPECTRA training lifecycle.
    """

    deterministic = bool(cfg.train.get("deterministic", False))
    deterministic_warn_only = bool(cfg.train.get("deterministic_warn_only", False))
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    configure_reproducibility(
        deterministic=deterministic,
        warn_only=deterministic_warn_only,
    )

    artifact_dir = resolve_artifact_dir(cfg)

    logger.info(f"Workspace: {output_dir}")
    logger.info(f"Stable Artifact Dir: {artifact_dir}")
    logger.info(
        "Reproducibility: deterministic=%s warn_only=%s cublas=%s",
        deterministic,
        deterministic_warn_only,
        str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")),
    )
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")




    preflight_check(cfg, output_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)


    config_path = save_experiment_config(cfg, artifact_dir)
    logger.info(f"Config saved to: {config_path}")

    metadata_path = save_experiment_metadata(cfg, artifact_dir)
    logger.info(f"Metadata saved to: {metadata_path}")


    datamodule = SPECTRADataModule(cfg)

    # Build the optimization engine and module via Hydra instantiation.
    # B-PGS requires a specialized decoupled engine injected directly.
    method_name = cfg.get("method_name") or cfg.get("method", {}).get("name")
    if method_name == "bpgs":
        from spectra.engine.optimizers.bpgs import BPGSEngine
        engine = BPGSEngine()
    else:
        engine = instantiate(cfg.module.engine)
        
    model = instantiate(cfg.module, cfg=cfg, engine=engine, _recursive_=False)


    callbacks = [
        *build_checkpoints(cfg, artifact_dir),
        GradientHealthCallback(check_interval=50),
        LearningRateMonitor(logging_interval="step"),
        build_progress_bar(cfg),
    ]
    es_cb = build_early_stopping(cfg)
    if es_cb is not None:
        callbacks.append(es_cb)

    loggers = []
    method_name = cfg.get("method_name", cfg.get("method", {}).get("name", "unknown"))
    dataset_name = cfg.get("dataset_name", "unknown")
    run_id = stable_run_id(cfg)
    
    csv_logger = CSVLogger(
        save_dir=str(artifact_dir),
        name="csv_logs",
        version=run_id,
    )
    loggers.append(csv_logger)
    logger.info(f"CSV logger: {artifact_dir}/csv_logs/{run_id}")

    wandb_session = WandbSession(
        WandbSettings.from_logging_config(cfg.get("logging", {})),
        artifact_dir,
        run_id,
        name=str(cfg.get("run_name", "unnamed_run")),
        group=f"{dataset_name}-{method_name}",
        job_type="training",
        tags=(dataset_name, method_name),
        config_payload=OmegaConf.to_container(cfg, resolve=True),
        summary_payload={
            "artifact_dir": str(artifact_dir),
            "dataset_name": dataset_name,
            "method_name": method_name,
        },
    )
    wandb_logger = wandb_session.create_lightning_logger()
    if wandb_logger is not None:
        loggers.append(wandb_logger)
        logger.info(f"WandB logger: {wandb_session.local_dir}")


    if torch.cuda.is_available() and not deterministic:
        torch.backends.cudnn.benchmark = True

    gradient_clip_val = cfg.train.get("grad_clip", 1.0)
    if not getattr(model, "automatic_optimization", True):
        gradient_clip_val = None
        method_name_local = cfg.get("method_name") or cfg.get("method", {}).get("name", "unknown")
        logger.info(
            f"Manual optimization active (method={method_name_local}). "
            f"PL gradient clipping disabled; engine manages clipping."
        )

    has_tqdm_bar = any(
        isinstance(cb, TQDMProgressBar) for cb in callbacks
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp_find_unused_parameters_false" if torch.cuda.device_count() > 1 else "auto",
        precision=cfg.train.get("precision", "16-mixed"),
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.train.get("log_every_n_steps", 10),
        deterministic=False,
        benchmark=(False if deterministic else None),
        enable_checkpointing=cfg.train.get("save_ckpt", False),
        enable_progress_bar=has_tqdm_bar,
    )


    resolved_resume = resolve_resume_checkpoint(cfg, artifact_dir)
    ckpt_path = str(resolved_resume) if resolved_resume is not None else None
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    elif str(cfg.get("resume_from", "")).strip().lower() == "auto":
        logger.info("resume_from=auto requested but no last.ckpt found. Starting fresh.")
    
    logger.info(
        f"Starting training: "
        f"method={cfg.get('method_name', cfg.get('method', {}).get('name', '?'))}, "
        f"epochs={cfg.train.epochs}, "
        f"tasks={[t.name for t in cfg.tasks]}"
    )
    
    fit_started_at = datetime.utcnow()
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        fit_ended_at = datetime.utcnow()

        checkpoint_registry = {}
        for callback in callbacks:
            if hasattr(callback, "monitor") and hasattr(callback, "best_model_path"):
                checkpoint_registry[str(callback.monitor)] = {
                    "mode": getattr(callback, "mode", None),
                    "best_model_path": getattr(callback, "best_model_path", None) or None,
                    "best_model_score": (
                        float(callback.best_model_score.item())
                        if getattr(callback, "best_model_score", None) is not None
                        else None
                    ),
                }

        summary_payload = {
            "fit_started_at": fit_started_at.isoformat() + "Z",
            "fit_ended_at": fit_ended_at.isoformat() + "Z",
            "elapsed_seconds": (fit_ended_at - fit_started_at).total_seconds(),
            "stopped_epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "checkpoint_registry": checkpoint_registry,
        }
        run_summary_path = save_run_summary(cfg, artifact_dir, summary_payload)
        wandb_session.update_summary(summary_payload)
        logger.info(f"Run summary saved to: {run_summary_path}")
        logger.info("Training completed successfully.")
    finally:
        wandb_session.finish()
