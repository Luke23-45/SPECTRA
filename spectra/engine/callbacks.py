"""
spectra/engine/callbacks.py
----------------------------
Axe v3.2: High-Integrity Monitoring Callbacks.

Provides deep introspection into the spectral properties of latent 
manifolds and the optimization dynamics of multi-task weighing.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import pytorch_lightning as pl
from typing import Dict, Any

logger = logging.getLogger("spectra.callbacks")


class SpectralMonitoringCallback(pl.Callback):
    """
    Validates Mechanism 2: ALB Spectral Decoupling.
    
    Computes the FFT power spectrum of activations in the Planner 
    vs the Expert manifolds to verify that high-frequency signals 
    are being successfully routed to the Expert.
    """

    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not pl_module.use_alb:
            return
        
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # We monitor the global context vectors from the ALB
        # Note: These values were cached during the last validation batch
        if not hasattr(pl_module.alb, "last_planner_ctx"):
            return

        p_ctx = pl_module.alb.last_planner_ctx  # [B, D]
        e_ctx = pl_module.alb.last_expert_ctx   # [B, D]

        # Compute 1D FFT over the representation dimension
        p_fft = torch.fft.rfft(p_ctx, dim=-1).abs()
        e_fft = torch.fft.rfft(e_ctx, dim=-1).abs()

        # Split into Low-Freq and High-Freq bins (simple 50/50 split for monitoring)
        mid_bin = p_fft.shape[-1] // 2
        
        p_low = p_fft[..., :mid_bin].mean()
        p_high = p_fft[..., mid_bin:].mean()
        
        e_low = e_fft[..., :mid_bin].mean()
        e_high = e_fft[..., mid_bin:].mean()

        # Ratios
        p_ratio = p_high / (p_low + 1e-6)
        e_ratio = e_high / (e_low + 1e-6)

        pl_module.log("spectral/planner_hf_ratio", p_ratio, sync_dist=True)
        pl_module.log("spectral/expert_hf_ratio", e_ratio, sync_dist=True)
        pl_module.log("spectral/hf_divorce_index", e_ratio / (p_ratio + 1e-6), sync_dist=True)

        logger.info(
            f"[Spectral-Audit] Epoch {trainer.current_epoch}: "
            f"Planner HF={p_ratio:.4f}, Expert HF={e_ratio:.4f} "
            f"(Divorce Index: {e_ratio/(p_ratio+1e-6):.2f}x)"
        )


class NTKGradExplosionTracker(pl.Callback):
    """
    Detects SI-MTL sub-gradient flattening.
    Monitors the relative norms of the shared gradients per task.
    """

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ):
        if trainer.global_step % 100 != 0:
            return
            
        # This is primarily for PCGrad or B-PGS monitoring
        if hasattr(pl_module.weighter, "last_grad_norms"):
            norms = pl_module.weighter.last_grad_norms
            for name, norm in norms.items():
                pl_module.log(f"audit/grad_norm_{name}", norm, sync_dist=True)
            
            max_norm = max(norms.values())
            min_norm = min(norms.values())
            pl_module.log("audit/grad_imbalance_ratio", max_norm / (min_norm + 1e-8), sync_dist=True)
