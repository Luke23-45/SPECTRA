"""
spectra/utils/optimizer.py
--------------------------
Shared optimization factory for the active canonical method path.
"""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from spectra.engine.schedulers import get_cosine_schedule_with_warmup


def build_optimizer_and_scheduler(module: pl.LightningModule, cfg: DictConfig) -> Dict[str, Any]:
    """
    Build the optimizer/scheduler stack.

    Active policy:
      - same AdamW family for all network parameters
      - B-PGS uses a second optimizer for uncertainty parameters
      - the active canonical path does not use the old 10x theta LR schedule
    """
    decay_params = []
    no_decay_params = []
    unc_params = []

    is_bpgs = getattr(module, "is_bpgs", False)

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        if is_bpgs and "weighter" in name:
            unc_params.append(param)
            continue

        no_weight_decay_keywords = ["bias", "norm", "bn", "LayerNorm", "weighter", "log_vars", "theta"]
        if any(keyword in name for keyword in no_weight_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.train.get("weight_decay", 0.0001)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=module.trainer.estimated_stepping_batches,
        min_lr_ratio=cfg.train.get("min_lr", 1e-6) / cfg.train.lr,
    )

    if is_bpgs:
        lr_unc = cfg.train.lr
        min_lr_ratio_theta = cfg.train.get("min_lr", 1e-6) / cfg.train.lr

        opt_unc = torch.optim.AdamW(unc_params, lr=lr_unc, weight_decay=0.0)
        scheduler_unc = get_cosine_schedule_with_warmup(
            opt_unc,
            num_warmup_steps=cfg.train.warmup_steps,
            num_training_steps=module.trainer.estimated_stepping_batches,
            min_lr_ratio=min_lr_ratio_theta,
        )

        return [optimizer, opt_unc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_unc, "interval": "step"},
        ]

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
