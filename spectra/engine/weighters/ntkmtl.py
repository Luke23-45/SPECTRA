"""
spectra/engine/weighters/ntkmtl.py
----------------------------------
NTKMTL: Neural Tangent Kernel Based Multi-Task Learning.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from spectra.engine.weighters.base import BaseWeighter

logger = logging.getLogger("spectra.ntkmtl")

class NTKMTLWeighter(BaseWeighter):
    def __init__(self, num_tasks: int, update_interval: int = 100, ema_decay: float = 0.9, **kwargs):
        super().__init__(num_tasks)
        self._update_interval = update_interval
        self._ema_decay = ema_decay
        self.register_buffer("weights", torch.ones(num_tasks))
        self.register_buffer("spectral_energy", torch.ones(num_tasks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("loss_ema", torch.ones(num_tasks))

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

        if self.training and shared_params is not None and self.step_count > 0 and self.step_count % self._update_interval == 0:
            if losses.grad_fn is not None:
                self._update_ntk_weights(losses, shared_params, sync_ddp)

        with torch.no_grad():
            self.step_count.add_(1)

        total = (self.weights.detach() * losses).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"ntkmtl/weight_{i}"] = self.weights[i].item()
            metrics[f"ntkmtl/spectral_{i}"] = self.spectral_energy[i].item()

        return total, metrics

    def _update_ntk_weights(self, losses, shared_params, sync_ddp):
        norms = []
        for i in range(self.num_tasks):
            grads = torch.autograd.grad(losses[i], shared_params, retain_graph=True, allow_unused=True)
            total_norm = sum((g.detach().norm() ** 2) if g is not None else torch.tensor(0.0, device=losses.device) for g in grads)
            norms.append(total_norm)

        norms_t = torch.stack(norms).to(losses.dtype)
        if sync_ddp and dist.is_initialized():
            dist.all_reduce(norms_t, op=dist.ReduceOp.SUM)
            norms_t /= dist.get_world_size()

        with torch.no_grad():
            valid_mask = (norms_t > 1e-8).float()
            updated_energy = torch.lerp(self.spectral_energy, norms_t, 1.0 - self._ema_decay)
            self.spectral_energy.copy_(torch.where(valid_mask > 0.5, updated_energy, self.spectral_energy))
            safe_energy = self.spectral_energy.clamp(min=1e-4)
            inv_weights = 1.0 / safe_energy
            self.weights.copy_(inv_weights * self.num_tasks / inv_weights.sum())
