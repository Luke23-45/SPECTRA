"""
spectra/utils/seed.py
---------------------
Deterministic seeding for full reproducibility.

Sets all random seeds across Python, NumPy, and PyTorch to ensure
bit-exact reproducibility of experiments across runs.
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger("spectra.seed")


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables CUDA deterministic algorithms.
                       May reduce performance by ~5%.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True avoids crashing on non-deterministic scatter_add
        torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(f"[Seed] All RNGs seeded to {seed} (deterministic={deterministic})")


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker seeding for reproducibility.

    Each worker receives a unique but deterministic seed derived from
    the main process seed + worker ID.

    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(seed)
    random.seed(seed)
