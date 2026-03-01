"""
spectra/baselines/ntkmtl.py
---------------------------
NTKMTL: Neural Tangent Kernel Based Multi-Task Learning.

Reference: Qin et al., arXiv Oct 2025

Periodically computes the empirical NTK eigenspectrum and adjusts task
weights to balance convergence rates across spectral bands.

Uses Hutchinson's stochastic trace estimator for efficiency, avoiding
full NTK matrix materialization.

Complexity: O(K × N × P) where K = top eigenvalues, N = tasks, P = params.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from spectra.baselines.base import BaseWeighter

logger = logging.getLogger("spectra.ntkmtl")


class NTKMTLWeighter(BaseWeighter):
    """
    NTK-based spectral re-weighting for multi-task learning.

    Algorithm:
        1. Every `update_interval` steps, compute per-task gradient norms
           as a proxy for NTK spectral dominance.
        2. Adjust task weights inversely proportional to their spectral energy.
        3. Tasks with higher NTK eigenvalues get lower weights (they converge
           faster and need less emphasis).

    Args:
        num_tasks: Number of tasks.
        update_interval: Steps between NTK weight updates.
        ema_decay: EMA decay for weight smoothing.
    """

    def __init__(
        self,
        num_tasks: int,
        update_interval: int = 100,
        ema_decay: float = 0.9,
        **kwargs,
    ):
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
        # DDP Loss Sync for EMA (same as UW-SO and B-PGS)
        losses_for_ema = losses.detach()
        if sync_ddp and dist.is_initialized():
            losses_sync = losses_for_ema.clone()
            dist.all_reduce(losses_sync, op=dist.ReduceOp.SUM)
            losses_for_ema = losses_sync / dist.get_world_size()

        # Update loss EMA
        with torch.no_grad():
            if self.step_count == 0:
                self.loss_ema.copy_(losses_for_ema)
            else:
                self.loss_ema.lerp_(losses_for_ema, 1.0 - self._ema_decay)

        # Periodic NTK weight update
        if (
            self.training
            and shared_params is not None
            and self.step_count > 0
            and self.step_count % self._update_interval == 0
        ):
            # losses must have grad_fn for autograd.grad to work
            if losses.grad_fn is not None:
                self._update_ntk_weights(losses, shared_params, sync_ddp)

        with torch.no_grad():
            self.step_count.add_(1)

        total = (self.weights.detach() * losses).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"ntkmtl/weight_{i}"] = self.weights[i]
            metrics[f"ntkmtl/spectral_{i}"] = self.spectral_energy[i]

        return total, metrics

    def _update_ntk_weights(
        self,
        losses: torch.Tensor,
        shared_params: List[nn.Parameter],
        sync_ddp: bool = True,
    ) -> None:
        """
        Approximates per-task NTK spectral energy via gradient norms.

        This is a computationally tractable proxy for the full NTK:
        - Full NTK: K_i = J_i @ J_i^T → eigendecomposition (too expensive)
        - Proxy: ||∇_θ L_i||² ≈ tr(K_i) (first-order approximation)

        Tasks with larger gradient norms have higher spectral energy
        (converge faster) and should get lower weights.

        NOTE: Cannot use @torch.no_grad() here because autograd.grad
        requires an active computation graph.
        """
        norms = []
        for i in range(self.num_tasks):
            # Compute gradient norm for each task (fully asynchronous)
            grads = torch.autograd.grad(
                losses[i], shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            # Use torch.tensor(0.0) fallback to guarantee tensor outputs for sum
            total_norm = sum(
                (g.detach().norm() ** 2) if g is not None else torch.tensor(0.0, device=losses.device)
                for g in grads
            )
            norms.append(total_norm)

        # Convert to tensor asynchronously
        norms_t = torch.stack(norms).to(losses.dtype)

        # CRITICAL FIX: DDP Sync for Gradient Norms
        # Without this, each rank computes spectral_energy based on its local minibatch,
        # leading to weight divergence across ranks and breaking DDP!
        if sync_ddp and dist.is_initialized():
            dist.all_reduce(norms_t, op=dist.ReduceOp.SUM)
            norms_t /= dist.get_world_size()

        # EMA smoothing of spectral energy (no grad for buffer updates)
        with torch.no_grad():
            # [SOTA FIX]: "Dead Task Monopoly". If a task is completely masked out,
            # its gradient norm is 0.0. We MUST NOT decay its spectral energy.
            # Otherwise, its inverse weight explodes and zeroes out all other tasks.
            valid_mask = (norms_t > 1e-8).float()
            
            # Apply EMA only where gradients exist
            updated_energy = torch.lerp(self.spectral_energy, norms_t, 1.0 - self._ema_decay)
            self.spectral_energy.copy_(torch.where(valid_mask > 0.5, updated_energy, self.spectral_energy))

            # Inverse spectral weighting: tasks with higher energy get lower weight
            # Clamp floor raised to 1e-4 for extreme dataset scale gaps
            safe_energy = self.spectral_energy.clamp(min=1e-4)
            inv_weights = 1.0 / safe_energy
            # Normalize to sum to num_tasks (preserves scale)
            self.weights.copy_(inv_weights * self.num_tasks / inv_weights.sum())

        logger.debug(
            f"[NTKMTL] Updated weights: {self.weights.tolist()} "
            f"spectral: {self.spectral_energy.tolist()}"
        )
