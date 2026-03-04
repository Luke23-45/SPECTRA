"""
spectra/baselines/pcgrad.py
---------------------------
PCGrad: Projecting Conflicting Gradients for Multi-Task Learning.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)

PCGrad modifies GRADIENTS, not losses. When two tasks have conflicting
gradients (negative dot product), it projects one onto the normal plane
of the other. Complexity: O(N²) in the number of tasks.

IMPORTANT: PCGrad requires a fundamentally different backward pass.
The training engine must detect PCGrad and call backward_and_project()
instead of total_loss.backward().
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.baselines.base import BaseWeighter


class PCGradWeighter(BaseWeighter):
    """
    PCGrad: Gradient Surgery for Multi-Task Learning.

    Algorithm:
        1. Compute per-task gradients g_i for each task loss.
        2. For each pair (i, j): if g_i · g_j < 0, project g_i onto
           the normal plane of g_j.
        3. Sum projected gradients.

    Complexity: O(N² × P) where N = tasks, P = parameters.

    USAGE NOTE: This class requires special interaction with the training engine.
    The forward() method returns the unweighted sum of losses.
    The actual gradient modification happens in backward_and_project().

    Args:
        num_tasks: Number of tasks.
    """

    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(num_tasks)
        self.register_buffer("conflict_count", torch.zeros(num_tasks))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns unweighted sum. Actual gradient surgery happens in backward_and_project()."""
        total = losses.sum()
        metrics = {
            f"pcgrad/conflict_{i}": self.conflict_count[i]
            for i in range(self.num_tasks)
        }
        return total, metrics

    def backward_and_project(
        self,
        task_losses: List[torch.Tensor],
        shared_params: List[nn.Parameter],
        skip_backward: bool = False,
        scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Custom backward with gradient projection (replaces total_loss.backward()).
        
        Refined Implementation (Full-Vector Projection):
        - Follows Yu et al. (2020) by projecting on the full gradient vector.
        - Stochastic projection order to prevent task bias.
        - Summation (not averaging) to preserve effective LR.
        """
        num_tasks = len(task_losses)
        device = task_losses[0].device
        
        # 1. Compute per-task gradients for shared parameters
        task_grads = []
        for idx, loss in enumerate(task_losses):
            is_last = (idx == num_tasks - 1)
            grads = torch.autograd.grad(
                loss, shared_params,
                retain_graph=not is_last,
                allow_unused=True,
            )
            grads = [
                g.detach() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, shared_params)
            ]
            task_grads.append(grads)

        # 2. Flatten and Project
        conflict_counts = torch.zeros(num_tasks, device=device)
        
        with torch.no_grad():
            # Flatten all per-task gradients to 1D vectors
            flat_grads = [
                torch.cat([g.reshape(-1) for g in tg])
                for tg in task_grads
            ]

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
            
            # 3. Aggregation: Dynamic SUM
            final_flat = torch.stack(projected_flat).sum(dim=0)
            
            if scale != 1.0:
                final_flat *= scale

            # 4. Unflatten and Assign to parameter.grad
            offset = 0
            for param in shared_params:
                numel = param.numel()
                grad_slice = final_flat[offset: offset + numel].reshape(param.shape)
                if param.grad is None:
                    param.grad = grad_slice.clone()
                else:
                    param.grad.copy_(grad_slice)
                offset += numel

        # 5. Telemetry (Conflict EMA update)
        with torch.no_grad():
            self.conflict_count.lerp_(conflict_counts, 0.1)

        metrics = {
            f"pcgrad/conflict_{i}": conflict_counts[i].item()
            for i in range(num_tasks)
        }
        metrics["pcgrad/total_conflicts"] = conflict_counts.sum().item()
        return metrics

    def project_and_assign(
        self,
        task_grads: List[List[torch.Tensor]],
        shared_params: List[nn.Parameter],
    ) -> Dict[str, float]:
        """
        Perform gradient surgery on PRE-COMPUTED per-task gradients.

        Refined Implementation (Full-Vector Projection):
        - Follows Yu et al. (2020) by projecting on the full gradient vector.
        - Stochastic projection order to prevent task bias.
        - Summation (not averaging) to preserve effective LR.
        """
        num_tasks = len(task_grads)
        device = shared_params[0].device if shared_params else "cpu"
        conflict_counts = torch.zeros(num_tasks, device=device)

        with torch.no_grad():
            # 1. Flatten all per-task gradients to 1D vectors
            flat_grads = []
            for tg in task_grads:
                flat_grads.append(torch.cat([g.reshape(-1) for g in tg]))

            projected_flat = []
            for i in range(num_tasks):
                gi = flat_grads[i].clone()

                # Randomize conflict check order (SOTA: prevents task bias)
                indices = list(range(num_tasks))
                random.shuffle(indices)

                for j in indices:
                    if i == j:
                        continue
                    gj = flat_grads[j]
                    dot = torch.dot(gi, gj)

                    if dot < 0:
                        # Project gi onto normal plane of gj
                        norm_sq = torch.dot(gj, gj) + 1e-8
                        gi -= (dot / norm_sq) * gj
                        conflict_counts[i] += 1

                projected_flat.append(gi)

            # 2. Aggregation: Dynamic SUM
            final_flat = torch.stack(projected_flat).sum(dim=0)

            # 3. Unflatten and assign to parameter.grad
            offset = 0
            for param in shared_params:
                numel = param.numel()
                grad_slice = final_flat[offset: offset + numel].reshape(param.shape)
                if param.grad is None:
                    param.grad = grad_slice.clone()
                else:
                    param.grad.copy_(grad_slice)
                offset += numel

        # Update running conflict EMA
        with torch.no_grad():
            self.conflict_count.lerp_(conflict_counts, 0.1)

        metrics = {
            f"pcgrad/conflict_{i}": conflict_counts[i].item()
            for i in range(num_tasks)
        }
        metrics["pcgrad/total_conflicts"] = conflict_counts.sum().item()
        return metrics

