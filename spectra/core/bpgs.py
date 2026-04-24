"""
spectra/core/bpgs.py
--------------------
Canonical B-PGS implementation for the active research path.

This is the single active B-PGS method in the repo:
  - bounded log-variance chart
  - detached precision in the network flow
  - raw batch losses for the uncertainty flow
  - optional relative-loss invariance to remove absolute-scale bias
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
      J_unc = sum_i [0.5 * omega_i * detach(L_i_tilde) + 0.5 * s_i]

    where L_i_tilde is either raw detached loss (legacy) or a geometric-mean
    normalized detached loss when relative_loss_invariance=True.
    """

    def __init__(
        self,
        num_tasks: int,
        s_min: float = -10.0,
        s_max: float = 10.0,
        s_init: float = 0.0,
        eps_clip: float = 1e-8,
        relative_loss_invariance: bool = True,
        relative_loss_floor: float = 1e-8,
        relative_loss_mode: str = "max",
        **kwargs,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}.")

        self.num_tasks = num_tasks
        self.relative_loss_invariance = bool(relative_loss_invariance)
        self.relative_loss_floor = float(relative_loss_floor)
        self.relative_loss_mode = str(relative_loss_mode)

        self.register_buffer("s_min_v", torch.full((num_tasks,), s_min))
        self.register_buffer("s_max_v", torch.full((num_tasks,), s_max))

        theta_init = _safe_logit(s_init, s_min, s_max, eps_clip)
        self.theta = nn.Parameter(torch.full((num_tasks,), theta_init))

    def get_s(self) -> torch.Tensor:
        """Project theta to the bounded log-variance manifold."""
        return self.s_min_v + (self.s_max_v - self.s_min_v) * torch.sigmoid(self.theta)

    def _detached_uncertainty_losses(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        losses = torch.stack([loss.detach() for loss in raw_losses])
        if not self.relative_loss_invariance:
            return losses

        safe_losses = losses.clamp_min(self.relative_loss_floor)
        if self.relative_loss_mode == "geo":
            denom = torch.exp(torch.mean(torch.log(safe_losses)))
        elif self.relative_loss_mode == "mean":
            denom = safe_losses.mean()
        elif self.relative_loss_mode == "max":
            denom = safe_losses.max()
        elif self.relative_loss_mode == "raw":
            denom = torch.tensor(1.0, device=safe_losses.device, dtype=safe_losses.dtype)
        else:
            raise ValueError(f"Unknown relative_loss_mode: {self.relative_loss_mode}")

        return safe_losses / denom.clamp_min(self.relative_loss_floor)

    def network_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Base flow objective for network parameters."""
        s = self.get_s()
        precision = torch.exp(-s).detach()

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + 0.5 * precision[i] * loss
        return total_loss

    def uncertainty_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Fiber flow objective for uncertainty parameters."""
        s = self.get_s()
        precision = torch.exp(-s)
        detached_losses = self._detached_uncertainty_losses(raw_losses)

        total_loss = 0
        for i in range(self.num_tasks):
            total_loss = total_loss + 0.5 * precision[i] * detached_losses[i] + 0.5 * s[i]
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
            "bpgs/relative_loss_invariance": torch.tensor(float(self.relative_loss_invariance), device=weights.device),
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

        stats: Dict[str, float] = {
            "bpgs/relative_loss_invariance": float(self.relative_loss_invariance)
        }
        for i in range(self.num_tasks):
            stats[f"bpgs/log_var_{i}"] = s[i].item()
            stats[f"bpgs/weight_{i}"] = weights[i].item()
            stats[f"bpgs/theta_{i}"] = self.theta[i].item()
        return stats
