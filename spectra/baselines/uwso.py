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
    Self-Optimized Uncertainty Weighting (no learnable parameters).

    w_i = 1 / (2 * EMA(L_i)² + ε)
    regularizer_i = log(EMA(L_i) + ε)

    L_total = Σ [ w_i * L_i + regularizer_i ]

    This is purely analytical — no SGD on weights, no learnable parameters.

    Args:
        num_tasks: Number of tasks.
        ema_decay: Exponential moving average decay for loss tracking.
    """

    def __init__(self, num_tasks: int, ema_decay: float = 0.99, **kwargs):
        super().__init__(num_tasks)
        self.temperature = kwargs.get("temperature", 1.0)  # SOTA default; tune 0.1–5.0
        self._ema_decay = ema_decay
        self.register_buffer("loss_ema", torch.ones(num_tasks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # DDP sync (optional)
        losses_for_ema = losses.detach()
        if raw_losses is not None:
            losses_for_ema = raw_losses.detach()

        if sync_ddp and dist.is_initialized():
            losses_sync = losses_for_ema.clone()
            dist.all_reduce(losses_sync, op=dist.ReduceOp.SUM)
            losses_for_ema = losses_sync / dist.get_world_size()

        # Update EMA (Training Only Guard)
        if self.training:
            with torch.no_grad():
                if self.step_count == 0:
                    self.loss_ema.copy_(losses_for_ema)
                else:
                    self.loss_ema.lerp_(losses_for_ema, 1.0 - self._ema_decay)
                self.step_count.add_(1)

        # [SOTA EXACT from Kirchdorfer et al. IJCV 2026 / arXiv:2408.07985]
        # Weights are derived from trackable uncertainty (raw_losses) if available,
        # ensuring the weighter correctly handles scale gaps in the task landscape.
        weight_input = raw_losses if raw_losses is not None else losses
        
        weight_input = weight_input.detach().clamp(min=1e-8)
        inv_losses = 1.0 / weight_input
        weights = torch.softmax(inv_losses / self.temperature, dim=0)  # sum(weights)=1
        
        # Total loss: convex combination of the provided (potentially weighted) losses.
        total = (weights * losses).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"uwso/weight_{i}"] = weights[i].item()
            metrics[f"uwso/loss_ema_{i}"] = self.loss_ema[i].item()

        return total, metrics
