"""
spectra/core/bpgs.py
--------------------
Version 3: Stateless B-PGS with pure theta-based parameterization.

This version removes all batch-stat dependence from the uncertainty manifold:
  - s depends only on theta (no mu/sigma from batch)
  - learnable global temperature (tau) instead of CV-based
  - fixed initialization (s_init) instead of auto-calibration
  - no cached state (no last_mu/last_sigma)
  - classical detached weights for network flow
  - classical uncertainty objective

Active method definition:
  s_i = s_min + (s_max - s_min) * sigmoid(theta_i)
  omega_i = exp(-s_i)
  J_net = sum_i stopgrad(omega_i) * L_i
  J_unc = sum_i [0.5 * omega_i * detach(L_i) + 0.5 * s_i]

Optional learnable temperature:
  T = softplus(tau) + 0.1
  Applied only to network-side normalization (not to uncertainty manifold).
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


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
        s_min: float = -10.0,
        s_max: float = 10.0,
        s_init: float = 0.0,
        eps_clip: float = 1e-8,
        learnable_temperature: bool = False,
        theta_grad_scale: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}.")
        if s_min >= s_max:
            raise ValueError(f"s_min must be strictly less than s_max; got {s_min} >= {s_max}.")

        self.num_tasks = num_tasks
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.eps_clip = float(eps_clip)
        self.theta_grad_scale = float(theta_grad_scale)

        # Fixed bounds for pure theta-based parameterization
        self.register_buffer("s_min_v", torch.full((num_tasks,), s_min))
        self.register_buffer("s_max_v", torch.full((num_tasks,), s_max))

        # Initialize theta so s starts at s_init
        p = (s_init - s_min) / (s_max - s_min)
        p = max(1e-4, min(1.0 - 1e-4, p))
        theta_init = math.log(p / (1.0 - p))
        self.theta = nn.Parameter(torch.full((num_tasks,), theta_init))

        # Learnable global temperature (optional)
        self.learnable_temperature = bool(learnable_temperature)
        if learnable_temperature:
            self.tau = nn.Parameter(torch.tensor(0.0))  # T = softplus(tau) + 0.1
        else:
            self.register_buffer("tau", torch.tensor(0.0))



    def get_s(self, raw_losses=None) -> torch.Tensor:
        """
        Project theta to bounded log-variance manifold (pure parameterization).
        
        Note: raw_losses parameter is ignored (kept for backward compatibility).
        """
        theta_scaled = GradScale.apply(self.theta, self.theta_grad_scale)
        return self.s_min_v + (self.s_max_v - self.s_min_v) * torch.sigmoid(theta_scaled)


    def network_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Network objective with classical detached precision weights."""
        s = self.get_s()
        precision = torch.exp(-s).detach()

        # Optional: normalize to sum to 1 for competition
        if self.learnable_temperature:
            T = F.softplus(self.tau) + 0.1
            weights = torch.softmax(precision / T, dim=0)  # no detach: allow tau to learn
        else:
            weights = precision

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + weights[i] * loss
        return total_loss



    def uncertainty_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Classical uncertainty objective for the bounded split-update BPGS path."""
        s = self.get_s()
        precision = torch.exp(-s)

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
            precision = torch.exp(-s)

            if self.learnable_temperature:
                T = F.softplus(self.tau) + 0.1
                weights = torch.softmax(precision / T, dim=0)
            else:
                weights = precision

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
            precision = torch.exp(-s)

            if self.learnable_temperature:
                T = F.softplus(self.tau) + 0.1
                weights = torch.softmax(precision / T, dim=0)
            else:
                weights = precision

        stats: Dict[str, float] = {}
        for i in range(self.num_tasks):
            stats[f"bpgs/log_var_{i}"] = s[i].item()
            stats[f"bpgs/weight_{i}"] = weights[i].item()
            stats[f"bpgs/theta_{i}"] = self.theta[i].item()
        if self.learnable_temperature:
            stats["bpgs/temperature"] = T.item()
        return stats
