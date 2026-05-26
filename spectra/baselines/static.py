"""
Static equal weighting — lower bound baseline.

Simply sums task losses with equal (or predefined) weights.
No learnable parameters. Provides the performance floor.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.baselines.base import BaseWeighter


class StaticWeighter(BaseWeighter):
    """
    Uniform or predefined static task weighting.

    L_total = Σ w_i * L_i

    Args:
        num_tasks: Number of tasks.
        weights: Optional list of fixed weights. If None, uses 1.0 for each task.
    """

    def __init__(self, num_tasks: int, weights: Optional[List[float]] = None, **kwargs):
        super().__init__(num_tasks)
        if weights is not None:
            if len(weights) != num_tasks:
                raise ValueError(f"Expected {num_tasks} weights, got {len(weights)}")
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("weights", torch.ones(num_tasks, dtype=torch.float32))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = (self.weights * losses).sum()
        metrics = {f"static/weight_{i}": self.weights[i].item() for i in range(self.num_tasks)}
        return total, metrics
