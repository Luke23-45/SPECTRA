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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns unweighted sum. Actual gradient surgery happens in backward_and_project()."""
        total = losses.sum()
        metrics = {
            f"pcgrad/conflict_{i}": self.conflict_count[i].item()
            for i in range(self.num_tasks)
        }
        return total, metrics

    def backward_and_project(
        self,
        task_losses: List[torch.Tensor],
        shared_params: List[nn.Parameter],
    ) -> Dict[str, float]:
        """
        Custom backward with gradient projection (replaces total_loss.backward()).

        Args:
            task_losses: List of per-task scalar losses (each requires_grad=True).
            shared_params: List of shared backbone parameters to modify gradients for.

        Returns:
            metrics: Conflict statistics for logging.
        """
        # 1. Compute per-task gradients
        task_grads = []
        for loss in task_losses:
            grads = torch.autograd.grad(
                loss, shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            # Flatten all parameter grads into a single vector
            flat = torch.cat([
                g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
                for g, p in zip(grads, shared_params)
            ])
            task_grads.append(flat)

        # 2. Pairwise projection (random order as in the paper)
        conflict_counts = torch.zeros(self.num_tasks, device=task_losses[0].device)
        projected = []

        for i in range(self.num_tasks):
            gi = task_grads[i].clone()
            order = list(range(self.num_tasks))
            random.shuffle(order)

            for j in order:
                if i == j:
                    continue
                gj = task_grads[j]
                dot = gi.dot(gj)

                if dot < 0:
                    # Conflict! Project gi onto normal plane of gj
                    gi = gi - (dot / (gj.norm() ** 2 + 1e-8)) * gj
                    conflict_counts[i] += 1

            projected.append(gi)

        # 3. Average projected gradients
        final_grad = torch.stack(projected).mean(dim=0)

        # 3.5. DDP Synchronization — CRITICAL FIX!
        # `autograd.grad` circumvents DDP hooks. Without this manual all_reduce,
        # ranks will diverge instantly because they update with local projected gradients!
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(final_grad, op=torch.distributed.ReduceOp.SUM)
            final_grad /= torch.distributed.get_world_size()

        # 4. Assign gradients to parameters
        offset = 0
        for param in shared_params:
            numel = param.numel()
            if param.grad is None:
                param.grad = final_grad[offset:offset + numel].reshape(param.shape).clone()
            else:
                param.grad.copy_(final_grad[offset:offset + numel].reshape(param.shape))
            offset += numel

        # Update running conflict counts
        with torch.no_grad():
            self.conflict_count.lerp_(conflict_counts, 0.1)

        metrics = {
            f"pcgrad/conflict_{i}": conflict_counts[i].item()
            for i in range(self.num_tasks)
        }
        metrics["pcgrad/total_conflicts"] = conflict_counts.sum().item()

        return metrics
