"""
spectra/baselines/base.py
-------------------------
Abstract base class for all MTL weighting methods.

All weighters share the same forward() signature, enabling
config-driven method swapping with zero code changes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseWeighter(nn.Module, ABC):
    """
    Abstract base for all MTL weighting methods.

    Every weighter must implement forward() with this exact signature.
    This allows the training engine to swap methods via config.

    Args:
        num_tasks: Number of tasks to weight.
    """

    def __init__(self, num_tasks: int, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks

    @abstractmethod
    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss.

        Args:
            losses: [num_tasks] tensor of per-task losses.
            shared_params: Optional list of shared backbone parameters.
                           Required for gradient-based methods (PCGrad).
            sync_ddp: Whether to synchronize across DDP ranks.
            raw_losses: Optional per-task unweighted losses for spectral estimation.

        Returns:
            total_loss: Scalar loss for backpropagation.
            metrics: Dict of method-specific telemetry for logging.
        """
        raise NotImplementedError
