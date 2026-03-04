"""
spectra/utils/optimizer.py
--------------------------
Shared Optimization Factory for all Vertical Silos.
Ensures zero duplication and protects Bayesian weighters from weight decay collapse.
"""

from typing import Dict, Any
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from spectra.engine.schedulers import get_cosine_schedule_with_warmup

def build_optimizer_and_scheduler(module: pl.LightningModule, cfg: DictConfig) -> Dict[str, Any]:
    """
    Constructs the standard AdamW + Cosine Warmup scheduler stack.
    Crucially separates normalization parameters and Bayesian weighters from weight decay.
    For B-PGS, it provisions the required decoupling (opt_net, opt_unc).
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
        
        # CRITICAL: Weighter parameters MUST NOT get weight decay, otherwise
        # log_vars/theta will be regularized towards 0, destroying Bayesian learning!
        no_weight_decay_keywords = [
            "bias", "norm", "bn", "LayerNorm",
            "weighter", "log_vars", "theta"
        ]
        if any(nd in name for nd in no_weight_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.train.get("weight_decay", 0.0001)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.train.lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=module.trainer.estimated_stepping_batches,
        min_lr_ratio=cfg.train.get("min_lr", 1e-6) / cfg.train.lr,
    )

    if is_bpgs:
        # SOTA Requirement: Decoupled Uncertainty Optimizer
        lr_unc = cfg.method.get("lr_theta", cfg.train.lr)
        opt_unc = torch.optim.AdamW(unc_params, lr=lr_unc, weight_decay=0.0)
        
        # SOTA Requirement: B-PGS Uncertainty parameters CANNOT have a static LR!
        scheduler_unc = get_cosine_schedule_with_warmup(
            opt_unc,
            num_warmup_steps=cfg.train.warmup_steps,
            num_training_steps=module.trainer.estimated_stepping_batches,
            min_lr_ratio=cfg.train.get("min_lr", 1e-6) / lr_unc,
        )
        
        # Return list of optimizers and list of scheduler configs 
        # for PyTorch Lightning manual optimization.
        return [optimizer, opt_unc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_unc, "interval": "step"}
        ]

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
