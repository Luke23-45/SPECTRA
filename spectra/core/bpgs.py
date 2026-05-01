"""
spectra/core/bpgs.py
--------------------
Canonical B-PGS v2 implementation for the active research path.

Three root-cause fixes over v1:
  1. ANTI-KENDALL UNCERTAINTY: Replace m with m^(-κ) in the uncertainty
     objective. This shifts the equilibrium from s*=log(L) (which converges
     to UWSO) to s*=-κ·log(L) (which favors hard tasks). κ=0 recovers
     the old Kendall equilibrium; κ>0 provides the anti-Kendall incentive.

  2. UNNORMALIZED PRECISION WEIGHTING: Remove softmax from the network
     loss. Use J_net = Σ stopgrad(exp(-s_i)) * L_i instead of softmax.
     This restores Kendall's equal-gradient property (weight * loss = const)
     while the bounded chart prevents gradient explosion. With anti-Kendall,
     the effective gradient becomes L^(κ+1)/ΣL^κ — mild hard-task focus.

  3. SMALL κ (0.1): At κ=0.1, the gradient distribution is nearly
     balanced (like Kendall) but with a slight hard-task bias. This
     prevents starvation while still prioritizing difficult tasks.

Retained from v1:
  - bounded batch-adaptive log-variance chart
  - detached weights in the network flow
  - raw batch losses for the uncertainty flow (detached)
  - one-shot auto-calibration (updated for anti-Kendall target)
  - no EMA, no extra architecture, no external training decoration
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class BPGS(nn.Module):
    """
    Canonical B-PGS v2 weighting module.

    Active method definition:
      Z_i = tau_T * (2 * sigmoid(theta_i) - 1)
      s_i = mu(L) + sigma(L) * Z_i
      omega_i = exp(-s_i)                              [precision]
      J_net = sum_i stopgrad(omega_i) * L_i             [unnormalized]
      J_unc = sum_i [0.5 * omega_i * detach(L_i)^(-kappa) + 0.5 * s_i]  [anti-Kendall]

    v1→v2 changes:
      - Uncertainty objective: m → m^(-κ)
        Shifts equilibrium from s*=log(L)=UWSO to s*=-κ·log(L).
        At new equilibrium: omega_i* = L_i^κ.
      - Network loss: softmax(exp(-s)/T) → unnormalized exp(-s)
        Removes softmax competition that caused weight collapse.
        At equilibrium: J_net = Σ L_i^κ * L_i = Σ L_i^(κ+1).
        With κ=0.1, gradient distribution is nearly balanced (like Kendall)
        but with mild hard-task focus.
      - Auto-calibration: target s=log(L) → target s=-κ·log(L)
      - temperature parameter: REMOVED from network_loss (no softmax)
    """

    def __init__(
        self,
        num_tasks: int,
        eps_clip: float = 1e-8,
        temperature: float = 2.0,
        kappa: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}.")
        self.num_tasks = num_tasks
        self.eps_clip = float(eps_clip)
        self.temperature = float(temperature)
        self.kappa = float(kappa)

        self.topological_limit = math.sqrt(num_tasks - 1) + 0.1
        self.register_buffer("last_mu", torch.zeros(1))
        self.register_buffer("last_sigma", torch.ones(1))

        self.theta = nn.Parameter(torch.zeros(num_tasks))
        self._calibrated = False

    def auto_calibrate(self, raw_losses: List[torch.Tensor]) -> None:
        """One-shot finite inverse-chart initialization from the current batch.

        v2: Targets s = -κ·log(L) instead of s = log(L), matching the
        anti-Kendall equilibrium instead of the Kendall equilibrium.
        """
        with torch.no_grad():
            detached_losses = torch.stack([loss.detach() for loss in raw_losses])
            log_l = torch.log(detached_losses.clamp(min=self.eps_clip))
            mu = log_l.mean()
            sigma = log_l.std(unbiased=False).clamp(min=1e-4)

            for i, loss in enumerate(raw_losses):
                # v2: anti-Kendall target s* = -κ * log(L)
                optimal_s = -self.kappa * torch.log(loss.detach().clamp(min=self.eps_clip))
                z_target = (optimal_s - mu) / sigma
                z_bounded = torch.clamp(z_target, -self.topological_limit, self.topological_limit)
                sigmoid_target = (z_bounded + self.topological_limit) / (2 * self.topological_limit)
                sigmoid_target = torch.clamp(sigmoid_target, 1e-4, 1.0 - 1e-4)
                self.theta.data[i] = torch.log(sigmoid_target / (1.0 - sigmoid_target))

    def get_s(self, raw_losses=None) -> torch.Tensor:
        """Project theta into the active bounded batch-adaptive log-variance chart."""
        if raw_losses is not None:
            if isinstance(raw_losses, torch.Tensor):
                detached_losses = raw_losses.detach()
            else:
                detached_losses = torch.stack([loss.detach() for loss in raw_losses])
            log_l = torch.log(detached_losses.clamp(min=self.eps_clip))
            mu = log_l.mean()
            sigma = log_l.std(unbiased=False).clamp(min=1e-4)
            self.last_mu[0] = mu
            self.last_sigma[0] = sigma
        else:
            mu = self.last_mu[0]
            sigma = self.last_sigma[0]

        z = self.topological_limit * (2 * torch.sigmoid(self.theta) - 1.0)
        return mu + z * sigma

    def network_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Network objective using detached unnormalized precision weights.

        v2: J_net = Σ stopgrad(exp(-s_i)) * L_i — NO softmax.
        This restores Kendall's equal-gradient property while the bounded
        chart prevents gradient explosion. With anti-Kendall equilibrium,
        exp(-s*) = L^κ, so effective gradient per task = L^(κ+1).
        """
        s = self.get_s(raw_losses)
        precision = torch.exp(-s).detach()

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + precision[i] * loss
        return total_loss

    def uncertainty_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Uncertainty objective using anti-Kendall incentive with detached raw batch losses.

        v2: Replace m_i with m_i^(-κ). This shifts the equilibrium from
        s*=log(m) (which converges to UWSO) to s*=-κ·log(m) (which favors
        hard tasks). κ=0 recovers the old Kendall equilibrium.

        Derivation of new equilibrium:
          ∂J_unc/∂s_i = -0.5 * exp(-s_i) * m_i^(-κ) + 0.5 = 0
          → exp(-s_i*) * m_i^(-κ) = 1
          → s_i* = -κ * log(m_i)

        At this equilibrium, network loss gradient per task becomes:
          ∂J_net/∂w ∝ exp(-s_i*) * ∂L_i/∂w = L_i^κ * ∂L_i/∂w
        With κ=0.1, this gives nearly balanced gradients (like Kendall's
        weight*loss=0.5) but with mild hard-task focus. The bounded chart
        prevents gradient explosion that Kendall's unbounded log_vars suffer.
        """
        if getattr(self, "_calibrated", False) is False:
            self.auto_calibrate(raw_losses)
            self._calibrated = True

        s = self.get_s(raw_losses)
        precision = torch.exp(-s)

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            # v2: Anti-Kendall — m^(-κ) instead of m
            effective_m = loss.detach().clamp(min=self.eps_clip).pow(-self.kappa)
            total_loss = total_loss + 0.5 * precision[i] * effective_m + 0.5 * s[i]
        return total_loss

    def forward(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compatibility interface for validation and logging."""
        with torch.no_grad():
            s = self.get_s(losses)
            precision = torch.exp(-s)
            # Normalize for logging (actual network_loss uses unnormalized)
            equalized_weights = precision / precision.sum()
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
        """Return current B-PGS telemetry for logging."""
        with torch.no_grad():
            s = self.get_s()
            precision = torch.exp(-s)
            # Normalize for logging
            equalized_weights = precision / precision.sum()

        stats: Dict[str, float] = {}
        for i in range(self.num_tasks):
            stats[f"bpgs/log_var_{i}"] = s[i].item()
            stats[f"bpgs/weight_{i}"] = equalized_weights[i].item()
            stats[f"bpgs/theta_{i}"] = self.theta[i].item()
        stats["bpgs/kappa"] = self.kappa
        return stats
