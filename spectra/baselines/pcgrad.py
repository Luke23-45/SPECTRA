"""
spectra/baselines/pcgrad.py
---------------------------
Canonical PCGrad baseline.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.baselines.base import BaseWeighter


class PCGradWeighter(BaseWeighter):
    """Gradient surgery for multi-task learning (Yu et al. 2020)."""

    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(num_tasks)

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return losses.sum(), {}

    def project_and_assign(
        self,
        task_grads: List[List[torch.Tensor]],
        shared_params: List[nn.Parameter],
    ) -> Dict[str, float]:
        num_tasks = len(task_grads)
        device = shared_params[0].device if shared_params else "cpu"
        conflict_counts = torch.zeros(num_tasks, device=device)

        with torch.no_grad():
            flat_grads = [torch.cat([g.reshape(-1) for g in tg]) for tg in task_grads]
            projected_flat = []

            for i in range(num_tasks):
                gi = flat_grads[i].clone()
                indices = list(range(num_tasks))
                random.shuffle(indices)

                for j in indices:
                    if i == j:
                        continue
                    gj = flat_grads[j]
                    dot = torch.dot(gi, gj)
                    if dot < 0:
                        norm_sq = torch.dot(gj, gj) + 1e-8
                        gi -= (dot / norm_sq) * gj
                        conflict_counts[i] += 1

                projected_flat.append(gi)

            final_flat = torch.stack(projected_flat).sum(dim=0)

            offset = 0
            for param in shared_params:
                numel = param.numel()
                grad_slice = final_flat[offset: offset + numel].reshape(param.shape)
                if param.grad is None:
                    param.grad = grad_slice.clone()
                else:
                    param.grad.copy_(grad_slice)
                offset += numel

        metrics = {f"pcgrad/conflict_{i}": conflict_counts[i].item() for i in range(num_tasks)}
        metrics["pcgrad/total_conflicts"] = conflict_counts.sum().item()
        return metrics
