"""Kendall uncertainty-weighting baseline."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.baselines.base import BaseWeighter


class KendallWeighter(BaseWeighter):
    """Unconstrained homoscedastic uncertainty weighting (Kendall et al. 2018)."""

    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(num_tasks)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
        raw_losses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        weighted = 0.5 * torch.exp(-self.log_vars) * losses
        reg = 0.5 * self.log_vars
        total = (weighted + reg).sum()

        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"kendall/log_var_{i}"] = self.log_vars[i].item()
            metrics[f"kendall/weight_{i}"] = (0.5 * torch.exp(-self.log_vars[i])).item()
        return total, metrics
