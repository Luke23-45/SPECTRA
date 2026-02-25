"""
spectra/engine/schedulers.py
----------------------------
Learning rate schedulers for SPECTRA training.

Provides cosine-with-warmup (SOTA for transformer training) and
optional OneCycleLR wrapper.
"""

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Linear warmup → Cosine decay → min_lr floor.

    This is the SOTA choice for transformer / MTL training.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Steps for linear warmup from 0 → base_lr.
        num_training_steps: Total steps for the cosine decay schedule.
        min_lr_ratio: Minimum LR as a fraction of base_lr (e.g., 0.01 = 1%).
        last_epoch: Last completed epoch (for resumption).

    Returns:
        LambdaLR scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay after warmup
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(progress, 1.0)  # Clamp for safety
        # Cosine annealing from 1.0 → min_lr_ratio
        cosine_decay = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
