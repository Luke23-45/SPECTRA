"""
spectra/engine/weighters/kendall.py
----------------------------------
Kendall et al. 2018 — Uncertainty Weighting.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.engine.weighters.base import BaseWeighter


class KendallWeighter(BaseWeighter):
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(num_tasks)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        precision = torch.exp(-self.log_vars)
        total = (0.5 * precision * losses + 0.5 * self.log_vars).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"kendall/log_var_{i}"] = self.log_vars[i].item()
            metrics[f"kendall/weight_{i}"] = (0.5 * precision[i]).item()

        return total, metrics
