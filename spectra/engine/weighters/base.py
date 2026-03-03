"""
spectra/engine/weighters/base.py
--------------------------------
Abstract base class for all MTL weighting methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseWeighter(nn.Module, ABC):
    """
    Abstract base for all MTL weighting methods.
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError
