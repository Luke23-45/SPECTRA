"""
spectra/engine/weighters/static.py
----------------------------------
Static Equal Weighting baseline.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.engine.weighters.base import BaseWeighter


class StaticWeighter(BaseWeighter):
    def __init__(self, num_tasks: int, weights: Optional[List[float]] = None, **kwargs):
        super().__init__(num_tasks)
        if weights is not None:
            assert len(weights) == num_tasks, f"Expected {num_tasks} weights, got {len(weights)}"
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("weights", torch.ones(num_tasks, dtype=torch.float32) / num_tasks)

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = (self.weights * losses).sum()
        metrics = {f"static/weight_{i}": self.weights[i].item() for i in range(self.num_tasks)}
        return total, metrics
