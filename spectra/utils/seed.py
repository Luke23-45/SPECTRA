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


def configure_reproducibility(deterministic: bool = True, warn_only: bool = False) -> None:
    """
    Configure backend-level determinism controls.

    This should run before training starts so CUDA/cuDNN/cuBLAS behavior is
    locked before the first kernels that matter for training are launched.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.deterministic = False


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

    configure_reproducibility(deterministic=deterministic, warn_only=False)

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
