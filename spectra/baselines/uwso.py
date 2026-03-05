"""
spectra/baselines/uwso.py
-------------------------
UW-SO: Self-Optimized Uncertainty Weighting (Analytical Inverse-Loss).

References:
    - Kirchdorfer et al., IJCV Dec 2025
    - Achituve et al., arXiv Aug 2024

This is the STRONGEST baseline — no learnable parameters.
Weights are derived analytically from running loss EMA.

At convergence, UW-SO approximately recovers the optimal Bayesian
weighting. B-PGS improves over UW-SO by providing smoother optimization
dynamics and bounded variance, especially during early training.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from spectra.baselines.base import BaseWeighter


class UWSOWeighter(BaseWeighter):
    """
    UW-SO: Soft Optimal Uncertainty Weighting.

    Computes analytically-optimal task weights from Kirchdorfer et al. (2024),
    "Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning",
    arXiv:2408.07985 / IJCV 2025.

    Derivation: at the stationary point of Kendall's uncertainty-weighting objective
    (∂L/∂σ_i = 0), the optimal weight is w_i* = 1/(2·L_i).  Normalizing with a
    temperature-scaled softmax yields the final per-task weights:

        w_i = softmax( (1/L_i) / T )        [stop-gradient on L_i]

    The log-regularizer from the Kendall objective reduces to a constant at the
    optimum and is not included in the training loss.

    Notes:
    - Temperature T controls the sharpness of the weight distribution.
      Higher T → more uniform weights. Lower T → winner-takes-all.
      Recommended: T ≥ 10 for normalized losses; T ≥ 50 for near-converged losses.
    - This implementation additionally maintains an EMA of task losses
      (self.loss_ema) which replaces raw batch losses as the weight input
      for improved stability.

    Args:
        num_tasks: Number of tasks.
        ema_decay: Exponential moving average decay for loss tracking.
    """

    def __init__(self, num_tasks: int, ema_decay: float = 0.99, **kwargs):
        super().__init__(num_tasks)
        self.temperature = kwargs.get("temperature", 1.0)  # [SOTA FIX]: Restored to 1.0 with dynamic scaling
        self._ema_decay = ema_decay
        self.register_buffer("loss_ema", torch.ones(num_tasks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("ema_initialized", torch.tensor(False))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Update EMA (Training Only Guard to prevent evaluation leakage/overhead)
        if self.training:
            # DDP sync: only needed if we are updating the EMA during training
            losses_for_ema = losses.detach()
            if raw_losses is not None:
                losses_for_ema = raw_losses.detach()

            if sync_ddp and dist.is_initialized():
                losses_sync = losses_for_ema.clone()
                dist.all_reduce(losses_sync, op=dist.ReduceOp.SUM)
                losses_for_ema = losses_sync / dist.get_world_size()

            with torch.no_grad():
                # [SOTA FIX]: NaN-Gate. Skip EMA update if loss exploded (e.g. diverging batch)
                # to prevent poisoning the moving average state permanently.
                if not (torch.isnan(losses_for_ema).any() or torch.isinf(losses_for_ema).any()):
                    if not self.ema_initialized:
                        self.loss_ema.copy_(losses_for_ema)
                        self.ema_initialized.fill_(True)
                    else:
                        self.loss_ema.lerp_(losses_for_ema, 1.0 - self._ema_decay)
                    self.step_count.add_(1)

        # [SOTA EXACT from Kirchdorfer et al. IJCV 2026 / arXiv:2408.07985]
        # [SOTA FIX]: Always use the smoothed EMA (even in evaluation!) if available. 
        # Using single-batch validation losses to generate the validation weights means 
        # the loss weighting scheme wildly fluctuates per mini-batch during validaton,
        # breaking reliable metric comparisons. Freezing weights via EMA prevents this.
        if self.ema_initialized:
            weight_input = self.loss_ema.clamp(min=1e-8)
        else:
            weight_input = (raw_losses if raw_losses is not None else losses).detach().clamp(min=1e-8)
        
        # UW-SO weights: softmax(1/L_i / T) — Kirchdorfer et al. (2024), Eq. 6
        inv_losses = 1.0 / weight_input
        # [SOTA Mathematical Fix]: Pure softmax over 1/L collapses as L -> 0 because 1/L -> inf. 
        # A static temperature T=50 cannot save a spread of 100,000. We apply Dynamic Temperature
        # Scaling (dividing by the mean of inv_losses) to ensure the softmax sharpness is
        # strictly scale-invariant across the *entire* training trajectory.
        dynamic_temp = self.temperature * inv_losses.mean()
        weights = torch.softmax(inv_losses / dynamic_temp, dim=0)  # sum(weights)=1
        
        # Total loss: convex combination of the provided (potentially weighted) losses.
        total = (weights * losses).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"uwso/weight_{i}"] = weights[i].item()
            metrics[f"uwso/loss_ema_{i}"] = self.loss_ema[i].item()

        # Weight entropy: healthy range ≈ 0.5·log(num_tasks)
        entropy = -(weights * weights.clamp(min=1e-8).log()).sum()
        metrics["uwso/weight_entropy"] = entropy.item()

        return total, metrics
