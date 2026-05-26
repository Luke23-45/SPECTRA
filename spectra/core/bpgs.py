"""
B-PGS weighting with uncertainty geometry.

Canonical configuration (validated by ablation on NYUv2, 4-epoch sweep):
  - s_mode = batch_aware      (s = mu + z*sigma, z from theta)
  - init_mode = auto_calibrate (one-shot first-batch theta calibration)

This combination yields:
  angle=39.40, abs_rel=0.314, mIoU=0.0831, vL=2.970
vs. the naive baseline (stateless + fixed):
  angle=43.60, abs_rel=0.320, mIoU=0.0776, vL=3.080

Ablation overrides (s_mode=stateless, init_mode=fixed) are kept for
reproducibility but are not recommended for production use.
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
    B-PGS weighting module with uncertainty geometry.

    Shared objective:
      omega_i = exp(-s_i)
      J_net = sum_i stopgrad(omega_i) * L_i
      J_unc = sum_i [0.5 * omega_i * detach(L_i) + 0.5 * s_i]

    Canonical configuration (validated by ablation on NYUv2):
      - s_mode = `batch_aware` : s depends on theta plus current batch log-loss stats
      - init_mode = `auto_calibrate` : one-shot first-batch calibration

    Ablation overrides (for study only, not recommended):
      - s_mode = `stateless`   : s depends only on theta
      - init_mode = `fixed`    : initialize theta from s_init
    """

    def __init__(
        self,
        num_tasks: int,
        s_min: float = -10.0,
        s_max: float = 10.0,
        s_init: float = 0.0,
        eps_clip: float = 1e-8,
        s_mode: str = "batch_aware",
        init_mode: str = "auto_calibrate",
        theta_grad_scale: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}.")
        if s_min >= s_max:
            raise ValueError(f"s_min must be strictly less than s_max; got {s_min} >= {s_max}.")
        if eps_clip <= 0:
            raise ValueError(f"eps_clip must be positive; got {eps_clip}.")
        if theta_grad_scale <= 0:
            raise ValueError(f"theta_grad_scale must be positive; got {theta_grad_scale}.")

        self.num_tasks = num_tasks
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.s_init = float(s_init)
        self.eps_clip = float(eps_clip)
        self.theta_grad_scale = float(theta_grad_scale)
        self.s_mode = str(s_mode)
        self.init_mode = str(init_mode)

        valid_s_modes = {"stateless", "batch_aware"}
        valid_init_modes = {"fixed", "auto_calibrate"}
        if self.s_mode not in valid_s_modes:
            raise ValueError(f"Unknown s_mode={self.s_mode!r}. Expected one of {sorted(valid_s_modes)}.")
        if self.init_mode not in valid_init_modes:
            raise ValueError(
                f"Unknown init_mode={self.init_mode!r}. Expected one of {sorted(valid_init_modes)}."
            )
        if not (self.s_min <= self.s_init <= self.s_max):
            raise ValueError(
                f"s_init must lie in [s_min, s_max]; got s_init={self.s_init}, "
                f"bounds=({self.s_min}, {self.s_max})."
            )

        self.register_buffer("s_min_v", torch.full((num_tasks,), s_min))
        self.register_buffer("s_max_v", torch.full((num_tasks,), s_max))

        # State used only by the batch-aware path and telemetry when raw losses are unavailable.
        self.topological_limit = math.sqrt(num_tasks - 1) + 0.1
        self.register_buffer("last_mu", torch.zeros(1))
        self.register_buffer("last_sigma", torch.ones(1))

        theta_init = self._theta_from_s(self.s_init)
        self.theta = nn.Parameter(torch.full((num_tasks,), theta_init))
        self._calibrated = False

    def _theta_from_s(self, s_value: float) -> float:
        p = (s_value - self.s_min) / (self.s_max - self.s_min)
        p = max(1e-4, min(1.0 - 1e-4, p))
        return math.log(p / (1.0 - p))

    def _theta_scaled(self) -> torch.Tensor:
        return GradScale.apply(self.theta, self.theta_grad_scale)

    def _bounded_stateless_s(self) -> torch.Tensor:
        return self.s_min_v + (self.s_max_v - self.s_min_v) * torch.sigmoid(self._theta_scaled())

    def _extract_detached_losses(self, raw_losses: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(raw_losses, torch.Tensor):
            return raw_losses.detach()
        return torch.stack([loss.detach() for loss in raw_losses])

    def _update_batch_stats(self, raw_losses: List[torch.Tensor] | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        losses = self._extract_detached_losses(raw_losses)
        log_losses = torch.log(losses.clamp(min=self.eps_clip))
        mu = log_losses.mean()
        sigma = log_losses.std(unbiased=False).clamp(min=1e-4)
        self.last_mu[0] = mu
        self.last_sigma[0] = sigma
        return mu, sigma

    def auto_calibrate(self, raw_losses: List[torch.Tensor] | torch.Tensor) -> None:
        """
        One-shot legacy-style initialization from the first observed loss batch.
        """
        with torch.no_grad():
            mu, sigma = self._update_batch_stats(raw_losses)
            losses = self._extract_detached_losses(raw_losses)
            log_losses = torch.log(losses.clamp(min=self.eps_clip))

            for i in range(self.num_tasks):
                z_target = (log_losses[i] - mu) / sigma
                z_bounded = torch.clamp(z_target, -self.topological_limit, self.topological_limit)
                sig_val = (z_bounded + self.topological_limit) / (2.0 * self.topological_limit)
                sig_val = torch.clamp(sig_val, 1e-4, 1.0 - 1e-4)
                self.theta.data[i] = -torch.log(1.0 / sig_val - 1.0)

    def get_s(self, raw_losses=None) -> torch.Tensor:
        """
        Construct current log-variances according to `s_mode`.
        """
        if self.s_mode == "stateless":
            return self._bounded_stateless_s()

        theta_scaled = self._theta_scaled()
        z = self.topological_limit * (2.0 * torch.sigmoid(theta_scaled) - 1.0)

        if raw_losses is not None:
            mu, sigma = self._update_batch_stats(raw_losses)
        else:
            mu = self.last_mu[0]
            sigma = self.last_sigma[0]

        return mu + z * sigma

    def _maybe_auto_calibrate(self, raw_losses: List[torch.Tensor]) -> None:
        if self.init_mode == "auto_calibrate" and not self._calibrated:
            self.auto_calibrate(raw_losses)
            self._calibrated = True

    def network_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        L1-normalized precision weighting for network parameters.
        Weights sum to 1, preventing loss-scale drift and gradient clipping distortion.
        """
        self._maybe_auto_calibrate(raw_losses)
        s = self.get_s(raw_losses if self.s_mode == "batch_aware" else None)
        weights = torch.exp(-s)
        weights = (weights / weights.sum()).detach()

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + weights[i] * loss
        return total_loss

    def uncertainty_loss(self, raw_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Classical uncertainty objective.
        """
        self._maybe_auto_calibrate(raw_losses)
        s = self.get_s(raw_losses if self.s_mode == "batch_aware" else None)
        precision = torch.exp(-s)

        total_loss = 0
        for i, loss in enumerate(raw_losses):
            total_loss = total_loss + 0.5 * precision[i] * loss.detach() + 0.5 * s[i]
        return total_loss

    def forward(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compatibility interface for validation/logging.
        """
        with torch.no_grad():
            s = self.get_s(losses if self.s_mode == "batch_aware" else None)
            weights = torch.exp(-s)
            weights = weights / weights.sum()
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
        """
        Return current uncertainty statistics for logging.
        """
        with torch.no_grad():
            s = self.get_s()
            weights = torch.exp(-s)
            weights = weights / weights.sum()

        stats: Dict[str, float] = {
            "bpgs/s_mode_batch_aware": 1.0 if self.s_mode == "batch_aware" else 0.0,
            "bpgs/init_mode_auto_calibrate": 1.0 if self.init_mode == "auto_calibrate" else 0.0,
        }
        for i in range(self.num_tasks):
            stats[f"bpgs/log_var_{i}"] = s[i].item()
            stats[f"bpgs/weight_{i}"] = weights[i].item()
            stats[f"bpgs/theta_{i}"] = self.theta[i].item()
        stats["bpgs/batch_mu"] = self.last_mu[0].item()
        stats["bpgs/batch_sigma"] = self.last_sigma[0].item()
        return stats
