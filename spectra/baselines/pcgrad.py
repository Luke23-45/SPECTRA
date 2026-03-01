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
        skip_backward: bool = False,
        scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Custom backward with gradient projection (replaces total_loss.backward()).

        Args:
            task_losses: List of per-task scalar losses (each requires_grad=True).
            shared_params: List of shared backbone parameters to modify gradients for.
            skip_backward: If True, assumes total_loss.backward(retain_graph=True) 
                          was already called externally (e.g. by PTL manual_backward).
            scale: Multiplier for final gradients (used to match AMP scaling).

        Returns:
            metrics: Conflict statistics for logging.
        """
        # 0. Optional: Component-wise backward (usually for heads)
        if not skip_backward:
            total_loss = sum(task_losses)
            total_loss.backward(retain_graph=True)

        # 1. Compute per-task gradients specifically for the shared backbone
        # We must use torch.autograd.grad even if backward was called,
        # to get the isolated per-task components for projection.
        task_grads = []
        for idx, loss in enumerate(task_losses):
            # [SOTA Fix: Instant GC] Force PyTorch to instantly free the massive  
            # computational graph buffers on the final task, rather than waiting for Python's GC.
            is_last = (idx == len(task_losses) - 1)
            grads = torch.autograd.grad(
                loss, shared_params,
                retain_graph=not is_last,
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

                # [SOTA Async Patch] Replace blocking `if dot < 0:` with tensor ops
                is_conflict = (dot < 0)
                
                # Compute projection vector
                proj = (dot / (gj.norm() ** 2 + 1e-8)) * gj
                
                # Apply only if conflict (avoids CPU-GPU sync pipeline stall)
                gi = torch.where(is_conflict, gi - proj, gi)
                conflict_counts[i] += is_conflict.float()

            projected.append(gi)

        # 3. Average projected gradients
        final_grad = torch.stack(projected).mean(dim=0)

        # 3.5. DDP Synchronization
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(final_grad, op=torch.distributed.ReduceOp.SUM)
            final_grad /= torch.distributed.get_world_size()

        # 4. Assign gradients to parameters (with optional scaling)
        offset = 0
        for param in shared_params:
            numel = param.numel()
            g_slice = final_grad[offset:offset + numel].reshape(param.shape)
            
            # [Axe Scaling Patch] Apply scale factor to match AMP expectations
            if scale != 1.0:
                g_slice = g_slice * scale

            if param.grad is None:
                param.grad = g_slice.clone()
            else:
                param.grad.copy_(g_slice)
            offset += numel

        # Update running conflict counts (async)
        with torch.no_grad():
            self.conflict_count.lerp_(conflict_counts, 0.1)

        # [SOTA Async Patch] Removed .item() to prevent CPU-GPU blocking at epoch tail
        metrics = {
            f"pcgrad/conflict_{i}": conflict_counts[i]
            for i in range(self.num_tasks)
        }
        metrics["pcgrad/total_conflicts"] = conflict_counts.sum()

        return metrics
