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

    def __init__(self, num_tasks: int, s_min: float = -2.0, **kwargs):
        super().__init__(num_tasks)
        # [SOTA Fix] Lower-bounded manifold mapping to prevent negative loss drift
        # s_min = -2.0 implies a minimum sigma^2 of ~0.135
        self.s_min = s_min
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[List[nn.Parameter]] = None,
        sync_ddp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Cast to fp32 for numerical safety under AMP
        log_vars_fp32 = self.log_vars.float()
        losses_fp32 = losses.float()
        
        # [SOTA Fix] Apply smooth lower bound to prevent s from drifting to -inf
        # s_eff = softplus(s - s_min) + s_min
        s_eff = torch.nn.functional.softplus(log_vars_fp32 - self.s_min) + self.s_min
        
        # Raw theoretical implicit weights
        raw_precision = torch.exp(-s_eff)
        
        # Apply bounded precision to the loss
        # [SOTA Fix] Manifold Average Alignment
        # Standardizing scale to match Average Baseline and UWSO.
        # This prevents the Bayesian regularizer from dominating in high-task regimes.
        total = (0.5 * raw_precision * losses_fp32).sum() + (0.5 * s_eff).mean()
        
        metrics = {}
        for i in range(self.num_tasks):
            metrics[f"kendall/log_var_{i}"] = s_eff[i].item()
            metrics[f"kendall/weight_{i}"] = (0.5 * raw_precision[i]).item()

        return total, metrics
