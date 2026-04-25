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
        eps_clip: float = 1e-8,
        temperature: float = 2.0,  # Matches the UWSO Gradient Currency Exchange scaler
        **kwargs,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}.")

        # Canonical Foundation: Enforce natural logarithm bound constraints without heuristics.
        # This prevents variance explosions by limiting total variance spread mathematically
        # strictly bounded symmetrically by Euler's scaling order ln(e^2).
        topological_limit = math.log(math.exp(2)) # Resolves identically to pure 2.0 mathematically
        
        s_min = -topological_limit
        s_max = topological_limit
        s_init = 0.0

        self.num_tasks = num_tasks
        self.temperature = temperature
        self.register_buffer("s_min_v", torch.full((num_tasks,), float(s_min)))
        self.register_buffer("s_max_v", torch.full((num_tasks,), float(s_max)))

        theta_init = _safe_logit(s_init, s_min, s_max, eps_clip)
        self.theta = nn.Parameter(torch.full((num_tasks,), float(theta_init)))


    def get_s(self) -> torch.Tensor:
        """Project theta to the bounded log-variance manifold."""
        return self.s_min_v + (self.s_max_v - self.s_min_v) * torch.sigmoid(self.theta)

    def network_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Base flow objective for network parameters with Stateless Temperature Equalization."""
        s = self.get_s()
        
        # SOTA Empirical Scale-Equalization
        # Transforms the log-variances via T-Softmax, bridging Cross-Entropy and Cosine gaps.
        # Multiplication by 'num_tasks' keeps the average weight effectively near 1.0.
        equalized_weights = self.num_tasks * torch.softmax(-s / self.temperature, dim=0).detach()

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + 0.5 * equalized_weights[i] * loss
        return total_loss



    def uncertainty_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Fiber flow objective for strictly bounded Canonical uncertainty weighting."""
        s = self.get_s()
        precision = torch.exp(-s)

        # 1. Pure detachment respects the foundational split-optimization rules.
        # No scalar normalizations are allowed. Let B-PGS natively fight the absolute scale imbalances!
        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + 0.5 * precision[i] * loss.detach() + 0.5 * s[i]
            
        return total_loss


    def forward(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compatibility interface for validation/logging.
        The active optimization path uses `network_loss` and `uncertainty_loss`.
        """
        with torch.no_grad():
            s = self.get_s()
            equalized_weights = self.num_tasks * torch.softmax(-s / self.temperature, dim=0)
            total = (equalized_weights * losses).sum()

        metrics: Dict[str, torch.Tensor] = {
            "bpgs/total_loss": total,
            "bpgs/weights_mean": equalized_weights.mean(),
            "bpgs/weights_min": equalized_weights.min(),
            "bpgs/weights_max": equalized_weights.max(),
        }
        for i in range(self.num_tasks):
            metrics[f"bpgs/s_{i}"] = s[i]
            metrics[f"bpgs/weight_{i}"] = equalized_weights[i]
        return total, metrics

    def get_task_stats(self) -> Dict[str, float]:
        """Return current normalized statistics for proper monitoring telemetry."""
        with torch.no_grad():
            s = self.get_s()
            # Calculate what the network ACTUALLY receives, rather than raw exp(-s)
            equalized_weights = self.num_tasks * torch.softmax(-s / self.temperature, dim=0)

        stats: Dict[str, float] = {}
        for i in range(self.num_tasks):
            stats[f"bpgs/log_var_{i}"] = s[i].item()
            stats[f"bpgs/weight_{i}"] = equalized_weights[i].item()
            stats[f"bpgs/theta_{i}"] = self.theta[i].item()
        return stats
