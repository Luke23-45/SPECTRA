"""
spectra/engine/weighters/uwso.py
-------------------------------
UW-SO: Self-Optimized Uncertainty Weighting.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from spectra.engine.weighters.base import BaseWeighter


class UWSOWeighter(BaseWeighter):
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
        losses_for_ema = losses.detach()
        if sync_ddp and dist.is_initialized():
            losses_sync = losses_for_ema.clone()
            dist.all_reduce(losses_sync, op=dist.ReduceOp.SUM)
            losses_for_ema = losses_sync / dist.get_world_size()

        with torch.no_grad():
            if self.step_count == 0:
                self.loss_ema.copy_(losses_for_ema)
            else:
                self.loss_ema.lerp_(losses_for_ema, 1.0 - self._ema_decay)
            self.step_count.add_(1)

        ema_safe = self.loss_ema.detach().clamp(min=1e-6)
        weights = 1.0 / (2.0 * ema_safe ** 2 + 1e-8)
        regularizer = torch.log(ema_safe + 1e-8)

        total = (weights * losses + regularizer).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"uwso/weight_{i}"] = weights[i].item()
            metrics[f"uwso/loss_ema_{i}"] = self.loss_ema[i].item()

        return total, metrics
