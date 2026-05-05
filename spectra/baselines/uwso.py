"""
spectra/baselines/uwso.py
-------------------------
Canonical UW-SO baseline with fixed temperature and raw batch losses.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.baselines.base import BaseWeighter


class UWSOWeighter(BaseWeighter):
    """Analytical inverse-loss weighting with fixed temperature."""

    def __init__(self, num_tasks: int, temperature: float = 2.0, **kwargs):
        super().__init__(num_tasks)
        if temperature <= 0:
            raise ValueError(f"temperature must be positive; got {temperature}.")
        self.temperature = temperature

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        weight_input = (raw_losses if raw_losses is not None else losses).detach().clamp(min=1e-8)
        inv_losses = 1.0 / weight_input
        weights = torch.softmax(inv_losses / self.temperature, dim=0)
        total = (weights * losses).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"uwso/weight_{i}"] = weights[i].item()
            metrics[f"uwso/loss_{i}"] = weight_input[i].item()
        return total, metrics
