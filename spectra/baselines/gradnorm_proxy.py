"""
spectra/baselines/gradnorm_proxy.py
-----------------------------------
Active gradient-norm proxy baseline inspired by Qin et al. (2025).

This is intentionally named as a proxy baseline, not as a faithful NTK-MTL implementation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from spectra.baselines.base import BaseWeighter


class GradNormProxyWeighter(BaseWeighter):
    """Gradient-norm proxy for spectral re-weighting."""

    def __init__(self, num_tasks: int, update_interval: int = 100, ema_decay: float = 0.9, **kwargs):
        super().__init__(num_tasks)
        self._update_interval = update_interval
        self._ema_decay = float(ema_decay)
        self.register_buffer("weights", torch.ones(num_tasks))
        self.register_buffer("spectral_energy", torch.ones(num_tasks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if (
            self.training
            and shared_params is not None
            and self.step_count > 0
            and self.step_count % self._update_interval == 0
            and losses.grad_fn is not None
        ):
            ntk_input = raw_losses if raw_losses is not None else losses
            self._update_spectral_weights(ntk_input, shared_params, sync_ddp)

        if self.training:
            with torch.no_grad():
                self.step_count.add_(1)

        total = (self.weights.detach() * losses).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"gradnorm_proxy/weight_{i}"] = self.weights[i].item()
            metrics[f"gradnorm_proxy/spectral_{i}"] = self.spectral_energy[i].item()
        return total, metrics

    def _update_spectral_weights(
        self,
        losses: torch.Tensor,
        shared_params: List[nn.Parameter],
        sync_ddp: bool = True,
    ) -> None:
        norms = []
        for i in range(self.num_tasks):
            grads = torch.autograd.grad(
                losses[i],
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            total_norm = sum(
                (g.detach().norm() ** 2) if g is not None else torch.tensor(0.0, device=losses.device)
                for g in grads
            )
            norms.append(total_norm)

        norms_t = torch.stack(norms).to(torch.float32)
        if sync_ddp and dist.is_initialized():
            dist.all_reduce(norms_t, op=dist.ReduceOp.SUM)
            norms_t /= dist.get_world_size()

        with torch.no_grad():
            if self._ema_decay > 0.0:
                self.spectral_energy.lerp_(norms_t, 1.0 - self._ema_decay)
            else:
                self.spectral_energy.copy_(norms_t)
            safe_energy = self.spectral_energy.clamp(min=1e-4)
            inv_weights = 1.0 / safe_energy
            self.weights.copy_(inv_weights * self.num_tasks / inv_weights.sum())
