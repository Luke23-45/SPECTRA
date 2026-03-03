"""
spectra/engine/weighters/pcgrad.py
----------------------------------
PCGrad: Projecting Conflicting Gradients.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.engine.weighters.base import BaseWeighter


class PCGradWeighter(BaseWeighter):
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(num_tasks)
        self.register_buffer("conflict_count", torch.zeros(num_tasks))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = losses.sum()
        metrics = {f"pcgrad/conflict_{i}": self.conflict_count[i].item() for i in range(self.num_tasks)}
        return total, metrics

    def backward_and_project(self, task_losses, shared_params, skip_backward=False, scale=1.0):
        num_tasks = len(task_losses)
        task_grads = []
        for idx, loss in enumerate(task_losses):
            is_last = (idx == num_tasks - 1)
            grads = torch.autograd.grad(loss, shared_params, retain_graph=not is_last, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            task_grads.append(grads)

        conflict_counts = torch.zeros(num_tasks, device=task_losses[0].device)
        with torch.no_grad():
            for p_idx, param in enumerate(shared_params):
                param_task_grads = [tg[p_idx] for tg in task_grads]
                projected_grads = []
                for i in range(num_tasks):
                    gi = param_task_grads[i].clone()
                    indices = list(range(num_tasks))
                    random.shuffle(indices)
                    for j in indices:
                        if i == j: continue
                        gj = param_task_grads[j]
                        dot = torch.sum(gi * gj)
                        if dot < 0:
                            norm_sq = torch.sum(gj * gj) + 1e-8
                            gi -= (dot / norm_sq) * gj
                            conflict_counts[i] += 1
                    projected_grads.append(gi)
                
                final_grad = torch.stack(projected_grads).sum(dim=0)
                if scale != 1.0: final_grad *= scale
                if param.grad is None: param.grad = final_grad.clone()
                else: param.grad.copy_(final_grad)

        if torch.distributed.is_initialized():
            for p in shared_params:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                p.grad /= torch.distributed.get_world_size()

        with torch.no_grad():
            self.conflict_count.lerp_(conflict_counts, 0.1)

        metrics = {f"pcgrad/conflict_{i}": conflict_counts[i].item() for i in range(num_tasks)}
        metrics["pcgrad/total_conflicts"] = conflict_counts.sum().item()
        return metrics

    def project_and_assign(self, task_grads, shared_params):
        num_tasks = len(task_grads)
        device = shared_params[0].device if shared_params else "cpu"
        conflict_counts = torch.zeros(num_tasks, device=device)

        with torch.no_grad():
            for p_idx, param in enumerate(shared_params):
                param_task_grads = [tg[p_idx] for tg in task_grads]
                projected_grads = []
                for i in range(num_tasks):
                    gi = param_task_grads[i].clone()
                    indices = list(range(num_tasks))
                    random.shuffle(indices)
                    for j in indices:
                        if i == j: continue
                        gj = param_task_grads[j]
                        dot = torch.sum(gi * gj)
                        if dot < 0:
                            norm_sq = torch.sum(gj * gj) + 1e-8
                            gi -= (dot / norm_sq) * gj
                            conflict_counts[i] += 1
                    projected_grads.append(gi)

                final_grad = torch.stack(projected_grads).sum(dim=0)
                if param.grad is None: param.grad = final_grad.clone()
                else: param.grad.copy_(final_grad)

        if torch.distributed.is_initialized():
            for p in shared_params:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                p.grad /= torch.distributed.get_world_size()

        with torch.no_grad():
            self.conflict_count.lerp_(conflict_counts, 0.1)

        metrics = {f"pcgrad/conflict_{i}": conflict_counts[i].item() for i in range(num_tasks)}
        metrics["pcgrad/total_conflicts"] = conflict_counts.sum().item()
        return metrics
