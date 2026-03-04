"""
spectra/engine/callbacks.py
----------------------------
Training infrastructure callbacks for SPECTRA.

Provides:
    1. SpectralMonitoringCallback — validates ALB frequency decoupling via FFT
    2. GradientHealthCallback    — real gradient norm monitoring with spike alerts
       (replaces the former NTKGradExplosionTracker which was a silent no-op)
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

logger = logging.getLogger("spectra.callbacks")


class SpectralMonitoringCallback(pl.Callback):
    """
    Validates ALB Mechanism: Spectral Decoupling.

    Computes FFT power spectrum of the Planner vs Expert manifold
    activations (cached by ALB during the last validation batch) to
    verify high-frequency signals route to Expert.

    Only runs when use_alb=True. No-op otherwise.
    """

    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not getattr(pl_module, "use_alb", False):
            return

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        alb = getattr(pl_module, "alb", None)
        if alb is None:
            return

        p_ctx = getattr(alb, "last_planner_ctx", None)  # [B, D]
        e_ctx = getattr(alb, "last_expert_ctx",  None)  # [B, D]

        if p_ctx is None or e_ctx is None:
            return

        # Detach and ensure fp32 for FFT stability
        p_ctx = p_ctx.detach().float().cpu()
        e_ctx = e_ctx.detach().float().cpu()

        # 1D FFT over representation dimension (D axis)
        p_fft = torch.fft.rfft(p_ctx, dim=-1).abs()  # [B, D//2+1]
        e_fft = torch.fft.rfft(e_ctx, dim=-1).abs()

        mid_bin = p_fft.shape[-1] // 2
        p_low  = p_fft[..., :mid_bin].mean().item()
        p_high = p_fft[..., mid_bin:].mean().item()
        e_low  = e_fft[..., :mid_bin].mean().item()
        e_high = e_fft[..., mid_bin:].mean().item()

        p_hf_ratio = p_high / (p_low + 1e-8)
        e_hf_ratio = e_high / (e_low + 1e-8)
        divorce_index = e_hf_ratio / (p_hf_ratio + 1e-8)

        pl_module.log("spectral/planner_hf_ratio", p_hf_ratio, sync_dist=True)
        pl_module.log("spectral/expert_hf_ratio",  e_hf_ratio, sync_dist=True)
        pl_module.log("spectral/hf_divorce_index", divorce_index, sync_dist=True)

        logger.info(
            f"[Spectral-Audit] Epoch {trainer.current_epoch}: "
            f"Planner HF={p_hf_ratio:.4f}, Expert HF={e_hf_ratio:.4f} "
            f"(Divorce Index: {divorce_index:.2f}x)"
        )

        # Warn if expert is NOT capturing more high-frequency than planner
        if divorce_index < 1.0:
            logger.warning(
                f"[Spectral-Audit] DEGRADATION: Expert HF ratio ({e_hf_ratio:.3f}) "
                f"<= Planner HF ratio ({p_hf_ratio:.3f}). "
                f"ALB spectral separation not achieved. Check expert branch init."
            )


class GradientHealthCallback(pl.Callback):
    """
    Real-time gradient health monitoring for the shared backbone.

    Monitors:
        1. Backbone total gradient L2 norm (absolute signal strength)
        2. Per-group update-to-weight ratio (‖grad‖/‖param‖)
           Healthy range: ~0.001. <1e-4 = vanishing, >0.1 = exploding.
        3. Gradient spike detection (EMA-based): warns when current norm
           exceeds 5× the running EMA. This catches instability before NaN.

    Runs every `check_interval` optimizer steps. Zero overhead in between.

    Note: Uses on_before_optimizer_step (Lightning hook on Callback),
    which fires after backward() but before optimizer.step() — exactly
    when gradients are populated and can be read without side effects.
    """

    def __init__(self, check_interval: int = 50, spike_threshold: float = 5.0):
        """
        Args:
            check_interval: Log every N optimizer steps.
            spike_threshold: Log a warning when norm > spike_threshold × EMA.
        """
        super().__init__()
        self.check_interval  = check_interval
        self.spike_threshold = spike_threshold
        self._grad_norm_ema: float = -1.0  # Uninitialized
        self._warned_this_epoch = set()   # Track conditions to prevent terminal spam

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._warned_this_epoch.clear()

    def _track_grad_health(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Skip Step 0 and respect interval to avoid initialization noise
        if trainer.global_step == 0 or trainer.global_step % self.check_interval != 0:
            return

        backbone = getattr(pl_module, "backbone", None)
        if backbone is None:
            return

        # ── 1. Backbone total gradient norm ──────────────────────────
        squared_norms = []
        param_norms   = []
        for p in backbone.parameters():
            if p.grad is not None and p.requires_grad:
                gn = p.grad.data.norm(2).item()
                pn = p.data.norm(2).item()
                squared_norms.append(gn ** 2)
                if pn > 1e-8:
                    param_norms.append((gn, pn))

        if not squared_norms:
            return

        total_grad_norm = sum(squared_norms) ** 0.5
        # Set prog_bar=True for high-visibility telemetry
        pl_module.log("health/backbone_grad_norm", total_grad_norm, sync_dist=False, prog_bar=True)

        # ── 2. Mean update-to-weight ratio ──────────────────────────
        if param_norms:
            ratios = [gn / pn for gn, pn in param_norms]
            mean_grad_weight_ratio = sum(ratios) / len(ratios)
            
            # [SOTA Fix] Actual update scales with the learning rate!
            optimizers = getattr(pl_module, "optimizers", lambda: None)()
            if optimizers is not None:
                opts = optimizers if isinstance(optimizers, list) else [optimizers]
                current_lr = opts[0].param_groups[0].get("lr", 1e-3)
            else:
                current_lr = 1e-3
                
            true_update_ratio = mean_grad_weight_ratio * current_lr
            pl_module.log("health/update_weight_ratio", true_update_ratio, sync_dist=False, prog_bar=False)

            # Flag to W&B for EASY monitoring. Terminal warnings are RATE-LIMITED.
            if true_update_ratio < 1e-6:
                if "vanishing" not in self._warned_this_epoch:
                    pl_module.print(
                        f"[GradHealth] Step {trainer.global_step}: ratio {true_update_ratio:.2e} "
                        f"is very small — possible vanishing gradient. (Silencing further warnings this epoch)"
                    )
                    self._warned_this_epoch.add("vanishing")
            elif true_update_ratio > 0.1:
                if "exploding" not in self._warned_this_epoch:
                    pl_module.print(
                        f"[GradHealth] Step {trainer.global_step}: ratio {true_update_ratio:.2e} "
                        f"is large — possible exploding gradient. (Silencing further warnings this epoch)"
                    )
                    self._warned_this_epoch.add("exploding")

        # ── 3. Spike detection (EMA-based) ──────────────────────────
        if self._grad_norm_ema < 0:
            self._grad_norm_ema = total_grad_norm
        else:
            self._grad_norm_ema = 0.99 * self._grad_norm_ema + 0.01 * total_grad_norm

        spike_ratio = total_grad_norm / (self._grad_norm_ema + 1e-8)
        if spike_ratio > self.spike_threshold:
            if "spike" not in self._warned_this_epoch:
                pl_module.print(
                    f"[GradSpike] Step {trainer.global_step}: grad_norm={total_grad_norm:.4f} "
                    f"is {spike_ratio:.1f}x EMA. Watch for instability. (Silencing per-step)"
                )
                self._warned_this_epoch.add("spike")
            pl_module.log("health/grad_spike_ratio", spike_ratio, sync_dist=False)

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Any
    ):
        # Only use this hook for AUTOMATIC optimization
        if getattr(pl_module, "automatic_optimization", True):
            self._track_grad_health(trainer, pl_module)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ):
        # Only use this hook for MANUAL optimization (PCGrad)
        if not getattr(pl_module, "automatic_optimization", True):
            self._track_grad_health(trainer, pl_module)

        # ── 4. Weighter-specific health (B-PGS theta saturation) ────
        weighter = getattr(pl_module, "weighter", None)
        if weighter is not None and hasattr(weighter, "theta"):
            theta_abs_max = weighter.theta.data.abs().max().item()
            pl_module.log("health/bpgs_theta_max", theta_abs_max, sync_dist=False)
            if theta_abs_max > 10.0:
                logger.warning(
                    f"[GradHealth] B-PGS theta saturation: max|θ|={theta_abs_max:.2f}. "
                    f"Sigmoid is nearly flat here — B-PGS update is slowing."
                )


# Backward-compatible alias (train.py imports this name)
NTKGradExplosionTracker = GradientHealthCallback
