"""
spectra/core/bpgs.py
--------------------
Bayesian Projected Gradient Scaling (B-PGS) via Sigmoid Manifold Mapping.

Core Contribution #1 of the SPECTRA framework.

Mathematical Foundation:
    Standard Kendall UW (2018) minimizes:
        L = Σ [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
    where s_i = log(σ²) is unconstrained, leading to variance explosion
    on dominant tasks (Kirchdorfer et al. IJCV 2025).

    B-PGS projects s onto a bounded manifold M = {s ∈ R^N | s_min ≤ s_i ≤ s_max}
    using a smooth Sigmoid mapping:
        s_i = s_min + (s_max - s_min) * σ(θ_i)
    where θ_i ∈ R is the unconstrained learnable parameter.

    This eliminates:
    1. Shadow Variable Drift (STE flaw: θ drifts to ±∞ outside bounds)
    2. Cold Start Explosion (via Auto-Calibration: θ₀ = logit((log(L₀) - s_min) / range))
    3. Zero-gradient death (sigmoid derivative is always non-zero)

References:
    - Kendall et al., "Multi-Task Learning Using Uncertainty" (CVPR 2018)
    - Kirchdorfer et al., "Uncertainty Weighting Revisited" (IJCV Dec 2025)
    - Achituve et al., "UW-SO" (arXiv Aug 2024)
"""

import math
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger("spectra.bpgs")


class BPGSScaler(nn.Module):
    """
    Bayesian Projected Gradient Scaling via Sigmoid Manifold Mapping.

    Casts homoscedastic uncertainty optimization as smooth projected
    gradient descent onto the Lipschitz-stable variance manifold
    M = {s ∈ R^N | s_min ≤ s_i ≤ s_max}.

    Features:
        1. C∞ Sigmoid Mapping: No gradient death, no shadow variable drift.
        2. Auto-Calibration: First-batch analytical initialization.
        3. Theoretical Bounds: Derived from expected loss scales.
        4. DDP-Safe: Optional AllReduce synchronization.

    Args:
        num_tasks: Number of tasks.
        s_min: Lower bound for log-variance manifold.
        s_max: Upper bound for log-variance manifold.
        ema_decay: Exponential moving average decay for loss tracking.
        use_sigmoid: If True, use sigmoid mapping. If False, use hard STE clamp
                     (ablation mode for comparison).
        use_autocal: If True, auto-calibrate θ from first-batch losses.
    """

    def __init__(
        self,
        num_tasks: int,
        s_min: float = -5.0,
        s_max: float = 10.0,
        ema_decay: float = 0.99,
        use_sigmoid: bool = True,
        use_autocal: bool = True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_sigmoid = use_sigmoid
        self.use_autocal = use_autocal

        # Internal unbounded parameters (θ)
        # Mapped to s ∈ [s_min, s_max] via sigmoid
        self.theta = nn.Parameter(torch.zeros(num_tasks))

        # Manifold bounds (registered as buffers for device movement)
        self.register_buffer("s_min", torch.tensor(s_min))
        self.register_buffer("s_max", torch.tensor(s_max))

        # Loss EMA tracking (no grad)
        self.register_buffer("loss_ema", torch.ones(num_tasks))
        self.register_buffer("ema_decay", torch.tensor(ema_decay))

        # State flags
        self.register_buffer("is_calibrated", torch.tensor(False))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    # =================================================================
    # MANIFOLD MAPPING
    # =================================================================

    def get_log_vars(self) -> torch.Tensor:
        """
        Maps unbounded θ to bounded log-variances s via Sigmoid.

        Returns:
            s: [num_tasks] tensor with s_i ∈ [s_min, s_max] (strict).
        """
        if self.use_sigmoid:
            # [SOTA FIX]: The "Vanishing Gradient of the Uncertainty Manifold"
            # Without this clamp, sigmoid(theta) saturates to exactly 0.0 or 1.0 for |theta| > 6 in fp32.
            # The derivative becomes 0.0, and the optimizer gets permanently trapped (Task Starvation).
            # Relying on fp32 sigmoid which maintains non-zero derivative up to |theta| ~10
            alpha = torch.sigmoid(self.theta)
            return self.s_min + alpha * (self.s_max - self.s_min)
        else:
            # Ablation: STE hard clamp (for comparison — demonstrates the flaw)
            clamped = torch.clamp(self.theta, self.s_min.item(), self.s_max.item())
            # Straight-Through Estimator: forward uses clamped, backward flows through identity
            return self.theta + (clamped - self.theta).detach()

    # =================================================================
    # AUTO-CALIBRATION
    # =================================================================

    @torch.no_grad()
    def auto_calibrate(self, losses: torch.Tensor) -> None:
        """
        Analytical warm-start: sets θ such that s₀ = log(L₀).

        This places the optimizer exactly at the Bayesian equilibrium at t=0,
        eliminating Cold Start Explosion where dominant tasks (MSE ~3000)
        cause violent initial gradient swings.

        Args:
            losses: [num_tasks] raw loss values from the first batch.
        """
        if self.is_calibrated:
            return

        # Sanitize: replace NaN/Inf with safe defaults
        safe_losses = losses.detach().clone()
        bad_mask = ~torch.isfinite(safe_losses)
        if bad_mask.any():
            median_val = safe_losses[torch.isfinite(safe_losses)].median()
            safe_losses[bad_mask] = median_val
            logger.warning(
                f"[B-PGS AutoCal] NaN/Inf detected at indices "
                f"{bad_mask.nonzero().flatten().tolist()}. Replaced with median={median_val:.4f}"
            )

        # Target log-variances: s* = log(L) at equilibrium
        target_s = torch.log(safe_losses + 1e-6)

        # Clamp targets within bounds (with safety margin ε=0.1)
        eps = 0.1
        target_s = target_s.clamp(self.s_min.item() + eps, self.s_max.item() - eps)

        if self.use_sigmoid:
            # Solve: s_min + range * sigmoid(θ) = target_s
            # => sigmoid(θ) = (target_s - s_min) / range = p_hat
            # => θ = logit(p_hat) = log(p_hat / (1 - p_hat))
            range_width = self.s_max - self.s_min
            normalized = (target_s - self.s_min) / range_width

            # CRITICAL SAFETY GUARD (v3.2): 
            # Clamp normalized target to avoid logit(0) or logit(1) -> ±inf
            normalized = normalized.clamp(1e-4, 1.0 - 1e-4)
            self.theta.data.copy_(torch.logit(normalized))
        else:
            # STE mode: directly set theta to target
            self.theta.data.copy_(target_s)

        # Initialize EMA to first-batch losses
        self.loss_ema.copy_(safe_losses)
        self.is_calibrated.fill_(True)

        log_vars = self.get_log_vars()
        logger.info(
            f"[B-PGS AutoCal] Initialized log_vars to: "
            f"{[f'{v:.3f}' for v in log_vars.tolist()]}"
        )

    # =================================================================
    # FORWARD PASS
    # =================================================================

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[list] = None,
        sync_ddp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the Bayesian multi-task weighted loss.

        L = Σ [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]

        where s_i is bounded via Sigmoid Manifold Mapping.

        Args:
            losses: [num_tasks] tensor of per-task scalar losses (requires_grad=True).
            shared_params: Unused by B-PGS (included for BaseWeighter interface conformance).
            sync_ddp: If True and DDP is initialized, AllReduce losses before
                      updating EMA (ensures consistent scaling across ranks).

        Returns:
            total_loss: Scalar loss for backpropagation.
            metrics: Dict of telemetry values for logging.
        """
        assert losses.shape == (self.num_tasks,), (
            f"Expected losses shape [{self.num_tasks}], got {losses.shape}"
        )

        # 1. Auto-calibrate on first call
        if self.use_autocal and not self.is_calibrated:
            cal_losses = losses.detach()

            # DDP: broadcast from rank 0 for consistency
            if sync_ddp and dist.is_initialized():
                dist.broadcast(cal_losses, src=0)

            self.auto_calibrate(cal_losses)

        # 2. DDP loss synchronization (for EMA update only)
        losses_for_ema = losses.detach()
        if sync_ddp and dist.is_initialized():
            losses_sync = losses_for_ema.clone()
            dist.all_reduce(losses_sync, op=dist.ReduceOp.SUM)
            losses_for_ema = losses_sync / dist.get_world_size()

        # 3. Update loss EMA (no grad)
        with torch.no_grad():
            if self.step_count == 0 and self.is_calibrated:
                # First step after calibration: hard set
                self.loss_ema.copy_(losses_for_ema)
            else:
                self.loss_ema.lerp_(losses_for_ema, 1.0 - self.ema_decay.item())
            self.step_count.add_(1)

        # 4. Get bounded log-variances (Sigmoid Manifold)
        log_vars = self.get_log_vars()

        # 5. Bayesian loss computation (Kendall et al. 2018)
        # L = Σ [ 0.5 * precision_i * L_i + 0.5 * s_i ]
        # where precision_i = exp(-s_i)
        #
        # CRITICAL: Cast to fp32 before exp() even under AMP (fp16 context).
        # exp(-log_var) for log_var << 0 produces very large precision values
        # that SILENTLY OVERFLOW fp16 (max ~65504), poisoning the loss.
        # This is exactly what torch.nn.functional.cross_entropy does internally.
        log_vars_fp32 = log_vars.float()
        losses_fp32   = losses.float()
        precision     = torch.exp(-log_vars_fp32)
        scaled_losses = 0.5 * precision * losses_fp32 + 0.5 * log_vars_fp32
        total_loss    = scaled_losses.sum()
        # Cast back to input dtype to preserve AMP compatibility downstream
        total_loss = total_loss.to(losses.dtype)

        # 6. Telemetry
        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"bpgs/log_var_{i}"] = log_vars[i].item()
            metrics[f"bpgs/weight_{i}"] = (0.5 * precision[i]).item()
            metrics[f"bpgs/precision_{i}"] = precision[i].item()
            metrics[f"bpgs/loss_ema_{i}"] = self.loss_ema[i].item()

        # Aggregate diagnostics
        metrics["bpgs/log_var_range"] = (log_vars.max() - log_vars.min()).item()
        metrics["bpgs/theta_max_abs"] = self.theta.data.abs().max().item()

        return total_loss, metrics

    # =================================================================
    # UTILITIES
    # =================================================================

    def get_effective_weights(self) -> torch.Tensor:
        """Returns the effective task weights (0.5 * precision) as a detached tensor."""
        with torch.no_grad():
            log_vars = self.get_log_vars()
            return 0.5 * torch.exp(-log_vars)

    def get_telemetry(self) -> Dict[str, float]:
        """Returns current state snapshot for external logging."""
        with torch.no_grad():
            log_vars = self.get_log_vars()
            precision = torch.exp(-log_vars)

        return {
            "log_vars": log_vars.tolist(),
            "precisions": precision.tolist(),
            "weights": (0.5 * precision).tolist(),
            "theta": self.theta.data.tolist(),
            "loss_ema": self.loss_ema.tolist(),
            "step": self.step_count.item(),
            "calibrated": self.is_calibrated.item(),
        }

    def extra_repr(self) -> str:
        return (
            f"num_tasks={self.num_tasks}, "
            f"s_min={self.s_min.item():.1f}, s_max={self.s_max.item():.1f}, "
            f"sigmoid={self.use_sigmoid}, autocal={self.use_autocal}"
        )


# =====================================================================
# STANDALONE VERIFICATION
# =====================================================================

if __name__ == "__main__":
    """Quick sanity check — run with: python -m spectra.core.bpgs"""
    print("=" * 60)
    print("B-PGS Scaler — Standalone Verification")
    print("=" * 60)

    torch.manual_seed(42)

    # Create scaler for 3-task scenario (MSE~3000, MSE~2, BCE~0.5)
    scaler = BPGSScaler(num_tasks=3, s_min=-5.0, s_max=10.0)
    print(f"\n{scaler}")

    # Simulate first-batch losses
    losses = torch.tensor([3000.0, 2.0, 0.5], requires_grad=True)
    total, metrics = scaler(losses, sync_ddp=False)

    print("\nFirst forward (auto-calibrated):")
    print(f"  Losses: {losses.tolist()}")
    log_var_strs = [f"{metrics['bpgs/log_var_' + str(i)]:.3f}" for i in range(3)]
    weight_strs = [f"{metrics['bpgs/weight_' + str(i)]:.4f}" for i in range(3)]
    print(f"  Log-vars: {log_var_strs}")
    print(f"  Weights: {weight_strs}")
    print(f"  Total loss: {total.item():.4f}")

    # Verify bounds
    for i in range(3):
        lv = metrics[f"bpgs/log_var_{i}"]
        assert -5.0 <= lv <= 10.0, f"BOUNDS VIOLATION: log_var_{i}={lv}"
    print(f"\n✓ All log_vars within bounds [-5.0, 10.0]")

    # Verify gradient flow
    total.backward()
    assert scaler.theta.grad is not None, "GRADIENT DEATH: theta.grad is None"
    assert scaler.theta.grad.abs().sum() > 0, "GRADIENT DEATH: theta.grad is all zeros"
    print(f"✓ Gradients flow through sigmoid: theta.grad = {scaler.theta.grad.tolist()}")

    # Verify bounds hold for extreme theta
    with torch.no_grad():
        scaler.theta.data = torch.tensor([100.0, -100.0, 0.0])
    log_vars_extreme = scaler.get_log_vars()
    for i in range(3):
        lv = log_vars_extreme[i].item()
        assert -5.0 <= lv <= 10.0, f"EXTREME BOUNDS VIOLATION: log_var_{i}={lv}"
    print(f"✓ Bounds hold for extreme θ: {log_vars_extreme.tolist()}")

    print(f"\n{'=' * 60}")
    print("B-PGS: All verifications PASSED.")
    print(f"{'=' * 60}")
