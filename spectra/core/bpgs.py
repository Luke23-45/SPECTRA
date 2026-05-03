"""
spectra/core/bpgs.py
--------------------
Active B-PGS implementation for the paper-track research path.

This is the single active B-PGS method in the repo:
  - bounded log-variance chart
  - detached precision in the network flow
  - raw batch losses for the uncertainty flow
  - adaptive CV-based temperature or fixed temperature
  - optional one-shot auto-calibration
  - no EMA, no R_eps, no prior term

Adaptive Temperature:
  Default behavior uses adaptive temperature based on batch statistics:
    T = σ(log_L) / |μ(log_L)| × 2.0
  
  High loss dispersion (CV) → higher temperature (more smoothing)
  Low loss dispersion (CV) → lower temperature (sharper competition)
  
  For fixed temperature, pass temperature=<value> to __init__.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


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
        eps_clip: float = 1e-8,
        temperature: float | None = None,  # None = use adaptive CV-based
        auto_calibrate: bool = True,
        theta_grad_scale: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}.")
        self.num_tasks = num_tasks
        self.temperature = temperature  # None = use adaptive CV-based
        self.auto_calibrate_enabled = bool(auto_calibrate)
        self.theta_grad_scale = float(theta_grad_scale)
        self.eps_clip = float(eps_clip)

        # Bounded latent radius used by the active chart.
        self.topological_limit = math.sqrt(num_tasks - 1) + 0.1 
        self.register_buffer("last_mu", torch.zeros(1))
        self.register_buffer("last_sigma", torch.ones(1))
        
        self.theta = nn.Parameter(torch.zeros(num_tasks))
        self._calibrated = False

    def auto_calibrate(self, raw_losses: List[torch.Tensor]) -> None:
        """
        One-shot initialization that maps the first batch loss landscape into
        the bounded latent chart.
        """
        with torch.no_grad():
            detached_losses = torch.stack([l.detach() for l in raw_losses])
            log_L = torch.log(detached_losses.clamp(min=self.eps_clip))
            mu = log_L.mean()
            sigma = log_L.std(unbiased=False).clamp(min=1e-4)

            for i, l in enumerate(raw_losses):
                opt_s = torch.log(l.clamp(min=self.eps_clip))
                
                # Project to Z-space
                Z_target = (opt_s - mu) / sigma
                Z_bounded = torch.clamp(Z_target, -self.topological_limit, self.topological_limit)
                
                # Inverse sigmoid to calculate the exact theta
                sig_val = (Z_bounded + self.topological_limit) / (2 * self.topological_limit)
                sig_val = torch.clamp(sig_val, 1e-4, 1.0 - 1e-4)
                
                self.theta.data[i] = -torch.log(1.0 / sig_val - 1.0)


    def get_s(self, raw_losses=None) -> torch.Tensor:
        """Project theta to the bounded log-variance manifold using SVAM."""
        if raw_losses is not None:
            if isinstance(raw_losses, torch.Tensor):
                detached_losses = raw_losses.detach()
            else:
                detached_losses = torch.stack([l.detach() for l in raw_losses])
            log_L = torch.log(detached_losses.clamp(min=self.eps_clip))
            mu = log_L.mean()
            sigma = log_L.std(unbiased=False).clamp(min=1e-4)
            self.last_mu[0] = mu
            self.last_sigma[0] = sigma
        else:
            mu = self.last_mu[0]
            sigma = self.last_sigma[0]

        theta_scaled = GradScale.apply(self.theta, self.theta_grad_scale)
        Z = self.topological_limit * (2 * torch.sigmoid(theta_scaled) - 1.0)
        return mu + Z * sigma

    def _compute_temperature(self, raw_losses: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """
        Adaptive temperature based on coefficient of variation of log-losses.
        
        T = σ(log_L) / |μ(log_L)| × 2.0
        
        High CV (dispersed losses) → higher temperature (more smoothing)
        Low CV (similar losses) → lower temperature (sharper competition)
        """
        if raw_losses is None:
            mu = self.last_mu[0]
            sigma = self.last_sigma[0]
            cv = sigma / mu.abs().clamp(min=self.eps_clip)
            return torch.clamp(cv * 2.0, min=0.1, max=5.0)

        if isinstance(raw_losses, list):
            losses = torch.stack([l.detach() for l in raw_losses])
        else:
            losses = raw_losses.detach()
        
        log_L = torch.log(losses.clamp(min=self.eps_clip))
        mu = log_L.mean()
        sigma = log_L.std(unbiased=False)
        cv = sigma / mu.abs().clamp(min=self.eps_clip)
        
        # Scale to reasonable range [0.5, 5.0]
        T = torch.clamp(cv * 2.0, min=0.1, max=5.0)
        return T

    def network_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Network objective with detached temperature-softmax precision weights."""
        s = self.get_s(raw_losses)

        if self.temperature is None:
            T = self._compute_temperature(raw_losses)
        else:
            T = self.temperature
        equalized_weights = torch.softmax(torch.exp(-s) / T, dim=0).detach()

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + equalized_weights[i] * loss
        return total_loss



    def uncertainty_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """Uncertainty objective for the bounded split-update BPGS path."""
        if self.auto_calibrate_enabled and getattr(self, "_calibrated", False) is False:
            self.auto_calibrate(raw_losses)
            self._calibrated = True

        s = self.get_s(raw_losses)
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
            s = self.get_s(losses)
            if self.temperature is None:
                T = self._compute_temperature(losses)
            else:
                T = self.temperature
            equalized_weights = torch.softmax(torch.exp(-s) / T, dim=0)
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
            if self.temperature is None:
                T = self._compute_temperature(None)
            else:
                T = self.temperature
            equalized_weights = torch.softmax(torch.exp(-s) / T, dim=0)

        stats: Dict[str, float] = {}
        for i in range(self.num_tasks):
            stats[f"bpgs/log_var_{i}"] = s[i].item()
            stats[f"bpgs/weight_{i}"] = equalized_weights[i].item()
            stats[f"bpgs/theta_{i}"] = self.theta[i].item()
        return stats
