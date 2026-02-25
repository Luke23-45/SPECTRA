"""
spectra/baselines/kendall.py
----------------------------
Kendall et al. 2018 — Unconstrained Homoscedastic Uncertainty Weighting.

Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)

This is the canonical MTL uncertainty weighting baseline. log_vars are
UNCONSTRAINED (no bounds), which intentionally demonstrates the
variance explosion problem that B-PGS solves.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectra.baselines.base import BaseWeighter


class KendallWeighter(BaseWeighter):
    """
    Unconstrained homoscedastic uncertainty weighting (Kendall et al. 2018).

    L_total = Σ [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]

    where s_i = log(σ²) are UNCONSTRAINED learnable parameters.

    Known failure mode: On tasks with extreme scale gaps (e.g., MSE ~3000 vs
    BCE ~0.5), the optimizer drives s₀ toward log(3000) ≈ 8.0, causing
    precision explosion and effective gradient death for the dominant task.

    Args:
        num_tasks: Number of tasks.
    """

    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(num_tasks)
        # INTENTIONALLY unconstrained — this is the flaw we demonstrate
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
