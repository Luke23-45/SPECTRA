"""
spectra/core/bpgs.py
--------------------
Canonical B-PGS implementation for the active research path.

This is the single active B-PGS method in the repo:
  - bounded log-variance chart
  - detached precision in the network flow
  - raw batch losses for the uncertainty flow
  - no EMA, no R_eps, no auto-calibration, no prior term
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _safe_logit(s_init: float, s_min: float, s_max: float, eps_clip: float = 1e-8) -> float:
    """Safely compute the inverse initialization for the sigmoid chart."""
    if s_min >= s_max:
        raise ValueError(f"s_min must be strictly less than s_max; got {s_min} >= {s_max}.")
    if not (s_min <= s_init <= s_max):
        raise ValueError(
            f"s_init must lie in [s_min, s_max]; got s_init={s_init}, bounds=({s_min}, {s_max})."
        )
    if not (0.0 < eps_clip < 0.5):
        raise ValueError(f"eps_clip must lie in (0, 0.5); got {eps_clip}.")

    p = (s_init - s_min) / (s_max - s_min)
    p = max(eps_clip, min(1.0 - eps_clip, p))
    return math.log(p / (1.0 - p))


class BPGS(nn.Module):
    """
    Canonical B-PGS weighting module.

    Active method definition:
      s_i = a_i + (b_i - a_i) * sigmoid(theta_i)
      omega_i = exp(-s_i)
      J_net = sum_i 0.5 * stopgrad(omega_i) * L_i
      J_unc = sum_i [0.5 * omega_i * detach(L_i) + 0.5 * s_i]
    """
    def __init__(
        self,
        num_tasks: int,
        omega_min: float = 0.1,    # Replaces s_max (Minimum weight limit)
        omega_max: float = 10.0,   # Replaces s_min (Maximum weight limit)
        omega_init: float = 1.0,   # Replaces s_init (Starting weight)
        eps_clip: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}.")
        if not (0.0 < omega_min < omega_max):
            raise ValueError("Precision bounds must satisfy 0 < omega_min < omega_max")
        if not (omega_min <= omega_init <= omega_max):
            raise ValueError("omega_init must be strictly within [omega_min, omega_max]")

        # Math Phase 1: Map Precision Space -> Log-Variance Space (s = -log(omega))
        # Note the inversion: Maximum precision corresponds to minimum log-variance.
        s_min = -math.log(omega_max)
        s_max = -math.log(omega_min)
        s_init = -math.log(omega_init)

        self.num_tasks = num_tasks
        self.register_buffer("s_min_v", torch.full((num_tasks,), float(s_min)))
        self.register_buffer("s_max_v", torch.full((num_tasks,), float(s_max)))

        theta_init = _safe_logit(s_init, s_min, s_max, eps_clip)
        self.theta = nn.Parameter(torch.full((num_tasks,), float(theta_init)))

    def get_s(self) -> torch.Tensor:
        """Project theta to the bounded log-variance manifold."""
        return self.s_min_v + (self.s_max_v - self.s_min_v) * torch.sigmoid(self.theta)

    def network_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Base flow objective for network parameters."""
        s = self.get_s()
        precision = torch.exp(-s).detach()

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + 0.5 * precision[i] * loss
        return total_loss



    def uncertainty_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Canonical Fiber flow objective for uncertainty parameters.
        Strict adherence to the formal B-PGS formulation.
        """
        s = self.get_s()
        precision = torch.exp(-s)

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            # Pure canonical surrogate: m_i = \ell_i(w)
            # Loss is detached to ensure split optimization paths (Step B)
            total_loss = total_loss + 0.5 * precision[i] * loss.detach() + 0.5 * s[i]
            
        return total_loss


    def forward(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compatibility interface for validation/logging.
        The active optimization path uses `network_loss` and `uncertainty_loss`.
        """
        with torch.no_grad():
            s = self.get_s()
            weights = torch.exp(-s)
            total = (weights * losses).sum()

        metrics: Dict[str, torch.Tensor] = {
            "bpgs/total_loss": total,
            "bpgs/weights_mean": weights.mean(),
            "bpgs/weights_min": weights.min(),
            "bpgs/weights_max": weights.max(),
        }
        for i in range(self.num_tasks):
            metrics[f"bpgs/s_{i}"] = s[i]
            metrics[f"bpgs/weight_{i}"] = weights[i]
        return total, metrics

    def get_task_stats(self) -> Dict[str, float]:
        """Return current uncertainty statistics for logging."""
        with torch.no_grad():
            s = self.get_s()
            weights = torch.exp(-s)

        stats: Dict[str, float] = {}
        for i in range(self.num_tasks):
            stats[f"bpgs/log_var_{i}"] = s[i].item()
            stats[f"bpgs/weight_{i}"] = weights[i].item()
            stats[f"bpgs/theta_{i}"] = self.theta[i].item()
        return stats
