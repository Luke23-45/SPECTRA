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
        self._ema_decay = ema_decay
        self.register_buffer("loss_ema", torch.ones(num_tasks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # DDP sync (optional)
        losses_for_ema = losses.detach()
        if sync_ddp and dist.is_initialized():
            losses_sync = losses_for_ema.clone()
            dist.all_reduce(losses_sync, op=dist.ReduceOp.SUM)
            losses_for_ema = losses_sync / dist.get_world_size()

        # Update EMA
        with torch.no_grad():
            if self.step_count == 0:
                self.loss_ema.copy_(losses_for_ema)
            else:
                self.loss_ema.lerp_(losses_for_ema, 1.0 - self._ema_decay)
            self.step_count.add_(1)

        # Analytical weights (NO learnable parameters)
        ema_safe = self.loss_ema.detach().clamp(min=1e-6)
        weights = 1.0 / (2.0 * ema_safe ** 2 + 1e-8)
        regularizer = torch.log(ema_safe + 1e-8)

        total = (weights * losses + regularizer).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"uwso/weight_{i}"] = weights[i].item()
            metrics[f"uwso/loss_ema_{i}"] = self.loss_ema[i].item()

        return total, metrics
