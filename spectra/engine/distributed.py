"""
spectra/engine/distributed.py
-----------------------------
DDP utilities for multi-GPU training.

Provides safe AllReduce and broadcast wrappers that gracefully
fall back to no-ops on single-GPU setups.
"""

import torch
import torch.distributed as dist
from typing import Optional


def is_distributed() -> bool:
    """Returns True if DDP is initialized and active."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Returns number of DDP processes (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    """Returns current process rank (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    """Returns True if current process is rank 0 (or single-GPU)."""
    return get_rank() == 0


def sync_losses(losses: torch.Tensor) -> torch.Tensor:
    """
    AllReduce losses across ranks and average.

    No-op if DDP not initialized.
    """
    if not is_distributed():
        return losses

    losses_sync = losses.clone()
    dist.all_reduce(losses_sync, op=dist.ReduceOp.SUM)
    return losses_sync / get_world_size()


def broadcast_from_rank0(tensor: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a tensor from rank 0 to all ranks.

    No-op if DDP not initialized.
    """
    if not is_distributed():
        return tensor

    dist.broadcast(tensor, src=0)
    return tensor
