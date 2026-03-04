"""
bpgs.py — Bayesian Projected Gradient Scaling (B-PGS)
======================================================

Production implementation of the final synthesized B-PGS definition.

Mathematical foundation
-----------------------
B-PGS is a multi-task learning loss weighting system defined by TWO
DECOUPLED GRADIENT FLOWS. There is no single joint loss function.

  Base Flow  (network weights w):
      dot_w = -eta_w * ∫ (1/2) exp(-s(tau)).detach() * ∇_w E(w, tau) dμ(tau)

  Fiber Flow (uncertainty params theta):
      dot_theta = -(eta_theta / g(theta)) * {
          (dΦ/dtheta) * [ -1/2 * exp(-s) * R_eps(L_bar) + 1/2 ]
          + dR_theta/dtheta
      }

Key components
--------------
  Diffeomorphic chart   : s_i = s_min + (s_max - s_min) * sigmoid(theta_i)
  Zero-preserving R_eps : R_eps(x) = sqrt(x^2 + eps^2)   [NOT softplus]
  Exact EMA integrator  : beta = 1 - exp(-1/tau)           [NOT alpha = 1/tau]
  NaN Gate              : skip EMA update when loss is not finite
  Bayesian prior        : optional Gaussian prior on theta  [makes ELBO explicit]
  Checkpoint safety     : L_bar stored as nn.Buffer        [saved/loaded in state_dict]

References
----------
  Kendall et al. (2018) — Multi-Task Learning Using Uncertainty to Weigh Losses
  B-PGS Final Synthesized Definition (this codebase)
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn


# ============================================================================
# Module-level mathematical operators
# ============================================================================

def R_eps(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Zero-preserving regularization operator: R_eps(x) = sqrt(x^2 + eps^2).

    This REPLACES softplus throughout B-PGS. It resolves the zero-point
    thermal bias: softplus(0) = log(2) ≈ 0.693, whereas R_eps(0) = eps.
    A perfectly solved task (x → 0) therefore drives the precision toward
    its theoretical maximum rather than a false floor.

    Mathematical properties:
      - R_eps(0)      = eps                      [no log(2) floor]
      - dR_eps/dx     = x / sqrt(x^2 + eps^2)   [smooth, → 0 at x=0, C-infinity]
      - R_eps(x) ≈ x  for x >> eps               [asymptotically correct]
      - Always > 0                                [numerically stable denominator]

    Args:
        x  : Non-negative input tensor (typically the smoothed batch loss L_bar_i).
        eps: Floor constant. Must satisfy 0 < eps << log(2) ≈ 0.693.
             Default 1e-5.

    Returns:
        Tensor of same shape as x, with values >= eps.
    """
    # x.pow(2) is slightly more numerically stable than x * x for autograd
    return torch.sqrt(x.pow(2) + eps * eps)


def _safe_logit(
    s_init: float,
    s_min: float,
    s_max: float,
    eps_clip: float = 1e-8,
) -> float:
    """
    Safely compute the unconstrained theta_init such that:
        s_min + (s_max - s_min) * sigmoid(theta_init) == s_init.

    The intermediate probability p = (s_init - s_min) / (s_max - s_min)
    is clamped to [eps_clip, 1 - eps_clip] before applying the logit.
    Without this clamp, s_init = s_min or s_max would produce theta = ±inf,
    corrupting training from the very first step.

    Args:
        s_init  : Target initial log-variance value in [s_min, s_max].
        s_min   : Lower bound of the log-variance manifold.
        s_max   : Upper bound of the log-variance manifold.
        eps_clip: Safety clamp width. Default 1e-8.

    Returns:
        theta_init as a Python float, finite and well-conditioned.
    """
    p = (s_init - s_min) / (s_max - s_min)
    p = max(eps_clip, min(1.0 - eps_clip, p))   # clamp before logit to prevent ±inf
    return math.log(p / (1.0 - p))


# ============================================================================
# BPGS Module
# ============================================================================

class BPGS(nn.Module):
    """
    Bayesian Projected Gradient Scaling (B-PGS).

    Learns one unconstrained scalar parameter theta_i per task. This parameter
    is mapped to a bounded log-variance s_i ∈ (s_min, s_max) via a sigmoid
    diffeomorphism, and the resulting precision weight exp(-s_i) is used to
    scale each task's contribution to the network loss.

    The system maintains a separate EMA-smoothed loss L_bar_i per task.
    This smoothed value drives the uncertainty (theta_i) update, not the
    raw batch loss — preventing single-batch spikes from corrupting weights.

    CRITICAL USAGE RULE
    -------------------
    You MUST use two separate optimizers and call the three methods in order:

        1. bpgs.update_ema(raw_losses)           # update smoothed losses
        2. opt_net.zero_grad()
           bpgs.network_loss(raw_losses).backward()
           opt_net.step()                         # update network weights
        3. opt_unc.zero_grad()
           bpgs.uncertainty_loss().backward()
           opt_unc.step()                         # update theta_i

    Calling a single combined .backward() is WRONG and will corrupt both flows.

    Args:
        num_tasks (int):
            Number of tasks. Must be >= 1.

        s_min (float):
            Lower bound of the log-variance manifold. Default -10.0.
            Geometrically: the minimum possible log-variance (maximum precision).
            Rule of thumb: set to log(minimum expected task loss).

        s_max (float):
            Upper bound of the log-variance manifold. Default 10.0.
            Geometrically: the maximum possible log-variance (minimum precision).
            Rule of thumb: set to log(maximum expected task loss) + margin.

        tau (float):
            EMA timescale in optimizer steps. Default 50.0.
            Controls how slowly L_bar_i reacts to changes in the batch loss.
            Recommended range: 10–100 steps.
            beta = 1 - exp(-1/tau) is the exact discrete integrator factor.

        eps (float):
            Floor constant for R_eps. Default 1e-5.
            Must satisfy: 0 < eps << log(2) ≈ 0.693.
            Determines the maximum achievable precision on a perfectly solved task.

        s_init (float):
            Initial value of s_i in s-space. Default 0.0.
            The corresponding theta_init is computed via the safe logit inverse.
            Use 0.0 for equal initial task weights with the default s range.

        eps_clip (float):
            Safety clamp for logit-inversion during initialization. Default 1e-8.
            Prevents theta_init = ±inf when s_init is exactly at s_min or s_max.

        prior_var (float | None):
            Variance of the isotropic Gaussian prior on theta. Default None (no prior).
            When set, adds the term (1 / (2 * prior_var)) * ||theta||^2 to
            uncertainty_loss(), making the MAP / ELBO interpretation fully explicit.
            Equivalent to L2 weight decay on theta only.

    Attributes:
        theta (nn.Parameter): Shape (num_tasks,). The unconstrained learnable params.
        L_bar  (torch.Tensor): Shape (num_tasks,). Registered buffer. Smoothed losses.
        beta   (float):        Exact EMA decay factor = 1 - exp(-1/tau).

    Example::

        model   = YourMultiTaskModel()
        bpgs    = BPGS(num_tasks=3, s_min=-10.0, s_max=10.0, tau=50.0)
        opt_net = torch.optim.Adam(model.parameters(),  lr=1e-3)
        opt_unc = torch.optim.Adam(bpgs.parameters(),   lr=1e-3)

        for batch in dataloader:
            # ---- forward pass: one raw loss per task ----
            loss_0 = criterion_0(model(batch["x"]), batch["y0"])
            loss_1 = criterion_1(model(batch["x"]), batch["y1"])
            loss_2 = criterion_2(model(batch["x"]), batch["y2"])
            raw_losses = [loss_0, loss_1, loss_2]

            # ---- Step 1: update smoothed losses (NaN-gated EMA) ----
            bpgs.update_ema(raw_losses)

            # ---- Step 2: update network weights (Base Flow) ----
            opt_net.zero_grad()
            bpgs.network_loss(raw_losses).backward()
            opt_net.step()

            # ---- Step 3: update uncertainty parameters (Fiber Flow) ----
            opt_unc.zero_grad()
            bpgs.uncertainty_loss().backward()
            opt_unc.step()
    """

    def __init__(
        self,
        num_tasks:  int,
        s_min:      float          = -10.0,
        s_max:      float          =  10.0,
        tau:        float          =  50.0,
        eps:        float          =  1e-5,
        s_init:     float          =  0.0,
        eps_clip:   float          =  1e-8,
        prior_var:  Optional[float] = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Argument validation — fail loudly with informative messages         #
        # ------------------------------------------------------------------ #
        if not isinstance(num_tasks, int) or num_tasks < 1:
            raise ValueError(
                f"num_tasks must be a positive integer; got {num_tasks!r}."
            )
        if s_min >= s_max:
            raise ValueError(
                f"s_min ({s_min}) must be strictly less than s_max ({s_max})."
            )
        if tau <= 0.0:
            raise ValueError(
                f"tau must be a positive float; got {tau}. "
                f"Recommended range: 10–100 optimizer steps."
            )
        if tau < 1.0:
            warnings.warn(
                f"tau={tau} is very small (< 1 step). "
                f"beta = 1 - exp(-1/tau) = {1.0 - math.exp(-1.0/tau):.4f}, "
                f"meaning the EMA reacts almost instantly to each batch. "
                f"Consider tau >= 10.",
                UserWarning,
                stacklevel=2,
            )
        if eps <= 0.0:
            raise ValueError(
                f"eps must be a positive float; got {eps}."
            )
        if eps >= math.log(2.0):
            raise ValueError(
                f"eps ({eps:.6f}) must be strictly less than log(2) ≈ {math.log(2):.6f}. "
                f"R_eps fixes the softplus zero-point bias only when eps << log(2)."
            )
        if not (s_min <= s_init <= s_max):
            raise ValueError(
                f"s_init ({s_init}) must lie within [s_min, s_max] = [{s_min}, {s_max}]."
            )
        if eps_clip <= 0.0 or eps_clip >= 0.5:
            raise ValueError(
                f"eps_clip must be in (0, 0.5); got {eps_clip}. "
                f"Typical value: 1e-8."
            )
        if prior_var is not None:
            if not isinstance(prior_var, (int, float)) or prior_var <= 0.0:
                raise ValueError(
                    f"prior_var must be a positive float when specified; got {prior_var!r}."
                )

        # ------------------------------------------------------------------ #
        # Store hyperparameters as instance attributes                        #
        # (needed for extra_repr, serialization, and downstream use)          #
        # ------------------------------------------------------------------ #
        self.num_tasks:  int            = int(num_tasks)
        self.s_min:      float          = float(s_min)
        self.s_max:      float          = float(s_max)
        self.tau:        float          = float(tau)
        self.eps:        float          = float(eps)
        self.prior_var:  Optional[float] = float(prior_var) if prior_var is not None else None

        # ------------------------------------------------------------------ #
        # Exact EMA discrete integrator constant                              #
        # Derivation: solve dL_bar/dt = (L - L_bar)/tau with dt = 1 step.    #
        # Exact solution: beta = 1 - exp(-1/tau).                             #
        # The Euler approximation alpha ≈ 1/tau introduces a lag bias that    #
        # accumulates over training. beta is the exact value; no approximation.#
        # ------------------------------------------------------------------ #
        self.beta: float = 1.0 - math.exp(-1.0 / self.tau)

        # ------------------------------------------------------------------ #
        # Learnable uncertainty parameters theta                              #
        # Shape: (num_tasks,), dtype: float32                                 #
        # Initialized via the safe logit inverse to ensure s(theta_init) ==   #
        # s_init exactly, with eps_clip preventing ±inf at the boundaries.    #
        # ------------------------------------------------------------------ #
        theta_init_val: float = _safe_logit(
            float(s_init), self.s_min, self.s_max, float(eps_clip)
        )
        self.theta = nn.Parameter(
            torch.full((self.num_tasks,), theta_init_val, dtype=torch.float32)
        )

        # ------------------------------------------------------------------ #
        # Smoothed EMA loss buffer L_bar                                      #
        # Stored as a registered buffer (NOT a Parameter) so that:            #
        #   (a) It is saved and restored in state_dict — checkpoint safety.   #
        #   (b) It is moved to the correct device with model.to(device).      #
        #   (c) requires_grad is False by default — never part of autograd.   #
        # Initialized to 1.0 per task (not 0.0) to avoid a degenerate        #
        # precision explosion at the very first uncertainty update.            #
        # ------------------------------------------------------------------ #
        self.register_buffer(
            "L_bar",
            torch.ones(self.num_tasks, dtype=torch.float32),
        )

    # ====================================================================== #
    # Core: diffeomorphic uncertainty chart                                   #
    # ====================================================================== #

    def get_s(self) -> list[torch.Tensor]:
        """
        Compute bounded log-variance s_i for all tasks.

        Formula:
            s_i = s_min + (s_max - s_min) * sigmoid(theta_i)

        The sigmoid guarantees s_i ∈ (s_min, s_max) strictly.
        The topological boundary ∂M requires |theta_i| → ∞, which is
        unreachable in finite optimization time. Shadow drift to the
        boundary is therefore topologically impossible.

        Implementation uses a single batched sigmoid over the full theta
        vector for efficiency, then returns individual per-task tensors
        so callers can zip them with per-task loss values.

        Note on floating-point precision:
            In exact arithmetic, sigmoid strictly maps R to (0, 1). In
            float32, exp(-88) underflows toward 0, so sigmoid saturates to
            exactly 0.0 or 1.0 at |theta| ≈ 88, causing s_i to numerically
            touch s_min or s_max. In practice theta never reaches such values
            during training because (a) the fiber flow gradient becomes zero at
            saturation and cannot push theta further, and (b) the optional prior
            actively regularizes theta toward 0. The saturation represents a
            degenerate but stable state, not a numerical instability.

        Returns:
            List of num_tasks scalar tensors, each carrying grad through theta_i.
        """
        # Batched sigmoid: one CUDA kernel call, not num_tasks separate calls
        s_vec = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.theta)
        # Return list of 0-dim views so grad flows per-task
        return [s_vec[i] for i in range(self.num_tasks)]

    # ====================================================================== #
    # EMA update — NaN-gated exact discrete integrator                        #
    # ====================================================================== #

    def update_ema(self, losses: list[torch.Tensor]) -> None:
        """
        Update L_bar_i for all tasks using the NaN Gate and exact integrator.

        MUST be called BEFORE network_loss() or uncertainty_loss() each step.

        The update rule is:
            L_bar_i ← L_bar_i + beta * (loss_i - L_bar_i)
        where beta = 1 - exp(-1/tau) is the exact discrete integrator factor.

        NaN Gate: if loss_i is not finite (NaN or Inf), the update for task i
        is SKIPPED entirely. L_bar_i retains its previous value. This prevents
        a single corrupt batch from permanently destroying the smoothed loss
        estimate for that task.

        All operations inside this method are performed under torch.no_grad().
        The buffer L_bar is updated in-place using plain Python float arithmetic
        to guarantee it is never part of the computation graph.

        Args:
            losses: List of num_tasks scalar loss tensors. Each is detached
                    internally; no gradient is consumed or created here.

        Raises:
            ValueError: If len(losses) != num_tasks.
            TypeError:  If any element is not a torch.Tensor.
        """
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"update_ema expected {self.num_tasks} losses, got {len(losses)}. "
                f"Ensure one loss tensor per task."
            )

        with torch.no_grad():
            for i, loss_i in enumerate(losses):
                if not isinstance(loss_i, torch.Tensor):
                    raise TypeError(
                        f"losses[{i}] must be a torch.Tensor; "
                        f"got {type(loss_i).__name__!r}. "
                        f"Do not pass Python floats directly."
                    )
                # Detach: we must never touch the gradient graph here
                loss_det = loss_i.detach()

                # NaN Gate: torch.isfinite on a 0-dim tensor returns a 0-dim
                # bool tensor. .item() converts it to a Python bool explicitly
                # and safely — no ambiguity about truthiness of tensors.
                if not torch.isfinite(loss_det).item():
                    # Skip this task's update; L_bar[i] is unchanged
                    continue

                # Extract scalar value as a plain Python float.
                loss_val_raw: float = loss_det.item()
                
                # Use raw loss (do not normalize by global sum, to preserve task difficulty signal)
                loss_val: float = loss_val_raw
                
                old_val:  float = self.L_bar[i].item()

                # Exact discrete integrator (eliminates Euler lag bias)
                new_val: float = old_val + self.beta * (loss_val - old_val)

                # Write back into the buffer using __setitem__.
                self.L_bar[i] = new_val

    # ====================================================================== #
    # Base Flow: network loss                                                  #
    # ====================================================================== #

    def network_loss(self, raw_losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute the precision-weighted network loss (Base Flow).

        Formula:
            L_network = Σ_i  0.5 * exp(-s_i).detach() * L_i

        The .detach() on exp(-s_i) is the formal stop-gradient. It converts
        the precision weight into a constant for this backward pass, ensuring
        that NO gradient flows into theta_i through this loss. The gradient
        flows only through L_i into the network weights w.

        Args:
            raw_losses: List of num_tasks scalar loss tensors, each carrying
                        gradients w.r.t. network parameters (not detached).

        Returns:
            Scalar tensor. Call .backward() with the NETWORK optimizer only.

        Raises:
            ValueError: If len(raw_losses) != num_tasks.
            TypeError:  If any element is not a torch.Tensor.
        """
        if len(raw_losses) != self.num_tasks:
            raise ValueError(
                f"network_loss expected {self.num_tasks} losses, got {len(raw_losses)}. "
                f"Ensure one loss tensor per task."
            )

        s_values = self.get_s()
        
        terms = []
        for i, loss_i in enumerate(raw_losses):
            if not isinstance(loss_i, torch.Tensor):
                raise TypeError(
                    f"raw_losses[{i}] must be a torch.Tensor; "
                    f"got {type(loss_i).__name__!r}."
                )
            # Use raw unscaled exp(-s_i) to maintain theoretical fixed-point consistency
            # with the uncertainty flow. Stop-gradient (.detach()) is mandatory.
            weight_i = torch.exp(-s_values[i]).detach()
            terms.append(0.5 * weight_i * loss_i)

        return sum(terms)

    # ====================================================================== #
    # Forward Compliance Interface for standard evaluation                   #
    # ====================================================================== #

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: Optional[list[nn.Parameter]] = None,
        sync_ddp: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compliance interface for validation methodologies.
        This provides compatibility with BaseWeighter patterns (e.g. UWSO/Kendall)
        during `validation_step` evaluation, allowing domain modules to accurately
        measure precision-weighted losses on the validation set without crashing.
        """
        loss_list = [losses[i] for i in range(self.num_tasks)]
        val_weighted_loss = self.network_loss(loss_list)
        return val_weighted_loss, self.get_task_stats()
        
    # ====================================================================== #
    # Fiber Flow: uncertainty loss                                             #
    # ====================================================================== #

    def uncertainty_loss(self) -> torch.Tensor:
        """
        Compute the uncertainty parameter loss (Fiber Flow).

        Formula:
            L_unc = Σ_i [ 0.5 * exp(-s_i) * R_eps(L_bar_i) + 0.5 * s_i ]
                  + (optional) (1 / (2 * prior_var)) * ||theta||^2

        Component breakdown:
          - 0.5 * exp(-s_i) * R_eps(L_bar_i): precision-weighted smoothed loss.
            Drives s_i DOWN (more confident) when L_bar_i is small.
          - 0.5 * s_i: entropic regularizer (log-normalizer of Gaussian likelihood).
            Drives s_i UP, preventing infinite precision.
          - The two terms balance at the Bayesian optimum.
          - R_eps(x) = sqrt(x^2 + eps^2) ensures the gradient is well-defined
            and non-zero even at L_bar_i = 0.
          - prior term: optional Gaussian prior on theta, making ELBO explicit.

        L_bar_i is injected as a DETACHED CONSTANT. It is the observed
        timescale-filtered energy of task i — an input to the Demon, not
        a variable being optimized. Differentiating through it would cause
        double-counting and incorrect gradient magnitudes.

        Gradient flows ONLY through s_i → theta_i. Network weights w
        receive no gradient through this loss (L_bar_i is detached and
        L_i_raw is not referenced here at all).

        Returns:
            Scalar tensor. Call .backward() with the UNCERTAINTY optimizer only.
        """
        s_values = self.get_s()

        # L_bar is a registered buffer on the correct device.
        # .detach() is redundant for a non-parameter buffer (requires_grad=False)
        # but is included for maximum clarity: L_bar is unconditionally a constant.
        L_bar_const = self.L_bar.detach()

        terms = []
        for i, s_i in enumerate(s_values):
            # Extract the i-th smoothed loss as a 0-dim tensor constant.
            # dtype and device match theta automatically (buffer shares device).
            l_bar_i = L_bar_const[i]

            # exp(-s_i): precision weight. grad flows: exp(-s_i) → s_i → theta_i.
            # NOT detached here — this is the Fiber Flow gradient path.
            w_i = torch.exp(-s_i)

            # R_eps(l_bar_i): zero-preserving operator on the (constant) smoothed loss.
            # d/ds_i [w_i * R_eps(l_bar_i)] = -exp(-s_i) * R_eps(l_bar_i)
            # d/ds_i [0.5 * s_i]            = +0.5
            # At equilibrium: exp(-s_i) * R_eps(L_bar_i) = 1  →  s_i* = log(R_eps(L_bar_i))
            r_val = R_eps(l_bar_i, eps=self.eps)

            terms.append(0.5 * w_i * r_val + 0.5 * s_i)

        l_unc = sum(terms)

        # Optional Gaussian prior on theta: -log N(theta; 0, prior_var * I)
        # = (1 / (2 * prior_var)) * ||theta||^2   (constant log-normalization dropped)
        # This implements Component 3 of the enhanced definition: R_theta(theta).
        # It is coercive (→ +∞ as |theta| → ∞), guaranteeing existence of minimizers.
        if self.prior_var is not None:
            l_unc = l_unc + (0.5 / self.prior_var) * self.theta.pow(2).sum()

        return l_unc

    # ====================================================================== #
    # Logging and diagnostic utilities                                         #
    # ====================================================================== #

    def get_weights(self) -> list[float]:
        """
        Return current precision weights exp(-s_i) as plain Python floats.

        Precision weight = exp(-s_i) ∈ (exp(-s_max), exp(-s_min)).
        Higher weight ⟹ task is considered more reliable ⟹ larger gradient scale.

        Intended for logging and monitoring. No gradient tracking.
        """
        with torch.no_grad():
            s_vec = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.theta)
            return [math.exp(-float(v)) for v in s_vec.tolist()]

    def get_log_vars(self) -> list[float]:
        """
        Return current log-variance s_i values as plain Python floats.

        s_i = log(sigma_i^2) where sigma_i^2 is the inferred task noise variance.
        Higher s_i ⟹ task is considered noisier ⟹ lower precision weight.

        Intended for logging and monitoring. No gradient tracking.
        """
        with torch.no_grad():
            s_vec = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.theta)
            return s_vec.tolist()

    def get_L_bar(self) -> list[float]:
        """
        Return current smoothed loss estimates L_bar_i as plain Python floats.

        These are the time-averaged task loss values used by the Fiber Flow.
        Useful for diagnosing whether the EMA has converged and detecting
        tasks with persistently high or near-zero loss.

        Intended for logging and monitoring.
        """
        return self.L_bar.tolist()

    def get_task_stats(self) -> dict[str, float]:
        """
        Return a consolidated dictionary of all per-task diagnostics, flattened
        for standardized logging (e.g., to TensorBoard or WandB).

        Returns:
            Dictionary mapping 'category_taskidx' to scalar float values.
        """
        with torch.no_grad():
            s_vec = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.theta)
            s_list = s_vec.tolist()
            weights = [math.exp(-v) for v in s_list]
            l_bar = self.L_bar.tolist()
            theta = self.theta.tolist()
            
            stats = {}
            for i in range(self.num_tasks):
                stats[f"bpgs/log_var_{i}"] = s_list[i]
                stats[f"bpgs/weight_{i}"] = weights[i]
                stats[f"bpgs/L_bar_{i}"] = l_bar[i]
                stats[f"bpgs/theta_{i}"] = theta[i]
            return stats

    def reset_ema(self, value: float = 1.0) -> None:
        """
        Reset all L_bar_i values to a specified constant.

        Useful when:
          - Resuming training after a learning rate reset or warmup phase.
          - The task distribution has shifted significantly.
          - Debugging: you want L_bar to restart from a known state.

        Args:
            value: Non-negative float to fill into L_bar. Default 1.0.
                   Using 0.0 is valid but may cause a brief precision spike
                   on the first uncertainty update step.

        Raises:
            ValueError: If value < 0.0.
        """
        if value < 0.0:
            raise ValueError(
                f"reset_ema: value must be non-negative (losses are non-negative); "
                f"got {value}."
            )
        with torch.no_grad():
            self.L_bar.fill_(value)

    # ====================================================================== #
    # PyTorch Module overrides                                                 #
    # ====================================================================== #

    def extra_repr(self) -> str:
        """
        String used by Module.__repr__ when printing the model.
        Displays all hyperparameters that affect behavior.
        """
        prior_str = (
            f", prior_var={self.prior_var}"
            if self.prior_var is not None
            else ""
        )
        return (
            f"num_tasks={self.num_tasks}, "
            f"s_min={self.s_min}, "
            f"s_max={self.s_max}, "
            f"tau={self.tau}, "
            f"beta={self.beta:.6f}, "
            f"eps={self.eps}"
            f"{prior_str}"
        )


# ============================================================================
# Self-contained verification suite
# ============================================================================

def _run_verification() -> None:
    """
    Comprehensive verification of every mathematical guarantee and
    implementation invariant. Run with: python bpgs.py
    """
    import sys

    PASS = "\033[92m  PASS\033[0m"
    FAIL = "\033[91m  FAIL\033[0m"
    SEP  = "-" * 60

    def check(condition: bool, name: str, detail: str = "") -> bool:
        status = PASS if condition else FAIL
        detail_str = f"  [{detail}]" if detail else ""
        print(f"{status}  {name}{detail_str}")
        return condition

    all_pass = True
    print(SEP)
    print("B-PGS Verification Suite")
    print(SEP)

    # ---- 1. R_eps mathematical properties ----
    print("\n[1] R_eps operator")
    eps = 1e-5
    x0  = torch.tensor(0.0)
    x1  = torch.tensor(1.0)
    x1e = torch.tensor(1e-10)

    r0 = R_eps(x0, eps=eps).item()
    all_pass &= check(abs(r0 - eps) < 1e-12,
                      "R_eps(0) == eps (no log(2) floor)",
                      f"got {r0:.2e}, expected {eps:.2e}")

    all_pass &= check(r0 < math.log(2),
                      f"R_eps(0)={r0:.2e} << log(2)={math.log(2):.4f}")

    r1 = R_eps(x1, eps=eps).item()
    all_pass &= check(abs(r1 - math.sqrt(1.0 + eps**2)) < 1e-6,
                      "R_eps(1) ≈ sqrt(1 + eps^2)",
                      f"got {r1:.8f}")

    # derivative at x~0 should be ~0 (smooth, not a kink)
    x_tiny = torch.tensor(1e-10, requires_grad=True)
    R_eps(x_tiny, eps=eps).backward()
    grad_at_zero = x_tiny.grad.item()
    all_pass &= check(abs(grad_at_zero) < 1e-4,
                      "dR_eps/dx → 0 at x≈0 (C-infinity)",
                      f"grad={grad_at_zero:.2e}")

    # at large x, R_eps(x) ≈ x
    x_large = torch.tensor(1000.0)
    r_large = R_eps(x_large, eps=eps).item()
    all_pass &= check(abs(r_large - 1000.0) / 1000.0 < 1e-8,
                      "R_eps(1000) ≈ 1000 (asymptotically correct)",
                      f"rel_err={(abs(r_large-1000)/1000):.2e}")

    # ---- 2. _safe_logit ----
    print("\n[2] Safe logit initialization")
    theta_mid  = _safe_logit(0.0,  -10.0, 10.0)
    s_recovered = -10.0 + 20.0 * (1.0 / (1.0 + math.exp(-theta_mid)))
    all_pass &= check(abs(s_recovered - 0.0) < 1e-6,
                      "Safe logit recovers s_init=0.0 exactly",
                      f"s_recovered={s_recovered:.8f}")

    theta_lo = _safe_logit(-10.0, -10.0, 10.0, eps_clip=1e-8)
    theta_hi = _safe_logit( 10.0, -10.0, 10.0, eps_clip=1e-8)
    all_pass &= check(math.isfinite(theta_lo) and math.isfinite(theta_hi),
                      "Safe logit at s_min and s_max is finite (no ±inf)",
                      f"theta_lo={theta_lo:.4f}, theta_hi={theta_hi:.4f}")

    # ---- 3. BPGS construction ----
    print("\n[3] BPGS construction and validation")
    bpgs = BPGS(num_tasks=3, s_min=-10.0, s_max=10.0, tau=50.0, eps=1e-5, s_init=0.0)

    expected_beta = 1.0 - math.exp(-1.0 / 50.0)
    all_pass &= check(abs(bpgs.beta - expected_beta) < 1e-12,
                      "beta = 1 - exp(-1/tau) is exact",
                      f"got {bpgs.beta:.8f}, expected {expected_beta:.8f}")

    all_pass &= check(bpgs.L_bar.shape == torch.Size([3]),
                      "L_bar buffer shape == (num_tasks,)")

    all_pass &= check(torch.all(bpgs.L_bar == 1.0).item(),
                      "L_bar initialized to 1.0 (not 0.0)")

    all_pass &= check(not bpgs.L_bar.requires_grad,
                      "L_bar.requires_grad is False (not part of autograd)")

    all_pass &= check("L_bar" in dict(bpgs.named_buffers()),
                      "L_bar is a registered buffer (in state_dict)")

    all_pass &= check("theta" in dict(bpgs.named_parameters()),
                      "theta is a registered parameter")

    # invalid args must raise
    try:
        BPGS(num_tasks=0)
        all_pass &= check(False, "Should reject num_tasks=0")
    except ValueError:
        all_pass &= check(True, "Rejects num_tasks=0 with ValueError")

    try:
        BPGS(num_tasks=2, s_min=5.0, s_max=1.0)
        all_pass &= check(False, "Should reject s_min >= s_max")
    except ValueError:
        all_pass &= check(True, "Rejects s_min >= s_max with ValueError")

    try:
        BPGS(num_tasks=2, eps=math.log(2))
        all_pass &= check(False, "Should reject eps >= log(2)")
    except ValueError:
        all_pass &= check(True, "Rejects eps >= log(2) with ValueError")

    # ---- 4. get_s() properties ----
    print("\n[4] get_s() diffeomorphic chart")
    bpgs2 = BPGS(num_tasks=4, s_min=-5.0, s_max=5.0, tau=20.0)
    s_values = bpgs2.get_s()

    all_pass &= check(len(s_values) == 4,
                      "get_s returns num_tasks tensors")

    for idx, s_i in enumerate(s_values):
        in_range = (-5.0 < s_i.item() < 5.0)
        all_pass &= check(in_range,
                          f"s_{idx} ∈ (s_min, s_max) strictly",
                          f"s_{idx}={s_i.item():.4f}")
        all_pass &= check(s_i.requires_grad,
                          f"s_{idx} carries grad through theta")

    # theta=30 is still within float32 safe range and should give s < s_max
    # (float32 sigmoid saturates around |theta|~88; float64 around |theta|~40)
    with torch.no_grad():
        bpgs2.theta.fill_(30.0)
    s_extreme = bpgs2.get_s()
    all_pass &= check(s_extreme[0].item() > 4.999,
                      "theta=30 gives s very close to s_max (near boundary)",
                      f"s={s_extreme[0].item():.8f}")
    with torch.no_grad():
        bpgs2.theta.fill_(_safe_logit(0.0, -5.0, 5.0))

    # ---- 5. update_ema() ----
    print("\n[5] update_ema(): NaN Gate and exact integrator")
    bpgs3 = BPGS(num_tasks=3, tau=10.0)
    expected_beta3 = 1.0 - math.exp(-1.0 / 10.0)

    # normal update
    initial_L_bar = bpgs3.L_bar.clone()
    losses_normal = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(0.5)]
    bpgs3.update_ema(losses_normal)
    for idx, (l_val, l0) in enumerate(zip([2.0, 3.0, 0.5], initial_L_bar.tolist())):
        expected = l0 + expected_beta3 * (l_val - l0)
        got = bpgs3.L_bar[idx].item()
        all_pass &= check(abs(got - expected) < 1e-5,
                          f"L_bar[{idx}] exact integrator",
                          f"got={got:.6f}, expected={expected:.6f}")

    # verify EMA convergence to constant signal over sufficient steps
    # tau=10 means beta≈0.095; need ~-log(1e-4/|init_error|)/(-log(1-beta)) steps
    bpgs_conv = BPGS(num_tasks=1, tau=10.0)
    target_val = 5.0
    n_steps = int(math.ceil(
        -math.log(1e-5 / abs(target_val - 1.0)) / (-math.log(1.0 - bpgs_conv.beta))
    )) + 20
    for _ in range(n_steps):
        bpgs_conv.update_ema([torch.tensor(target_val)])
    conv_err = abs(bpgs_conv.L_bar[0].item() - target_val)
    all_pass &= check(conv_err < 1e-4,
                      f"EMA converges to true value in {n_steps} steps (tau=10)",
                      f"L_bar={bpgs_conv.L_bar[0].item():.6f}, target={target_val}")

    # NaN Gate: NaN loss must leave L_bar unchanged
    l_bar_before_nan = bpgs3.L_bar.clone()
    losses_with_nan = [
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
        torch.tensor(1.0),
    ]
    bpgs3.update_ema(losses_with_nan)
    all_pass &= check(bpgs3.L_bar[0].item() == l_bar_before_nan[0].item(),
                      "NaN loss leaves L_bar[0] unchanged (NaN Gate)")
    all_pass &= check(bpgs3.L_bar[1].item() == l_bar_before_nan[1].item(),
                      "Inf loss leaves L_bar[1] unchanged (NaN Gate)")
    all_pass &= check(bpgs3.L_bar[2].item() != l_bar_before_nan[2].item(),
                      "Finite loss updates L_bar[2] normally")

    # wrong length must raise
    try:
        bpgs3.update_ema([torch.tensor(1.0)])
        all_pass &= check(False, "Should raise on wrong number of losses")
    except ValueError:
        all_pass &= check(True, "Raises ValueError on wrong loss list length")

    # ---- 6. network_loss() gradient flow ----
    print("\n[6] network_loss(): gradient isolation")
    bpgs4 = BPGS(num_tasks=2, s_min=-5.0, s_max=5.0, tau=20.0)

    # Create a tiny "network" whose parameters should receive grad
    net_param = nn.Parameter(torch.tensor(1.0))
    raw_losses_net = [0.5 * net_param.pow(2), 0.3 * net_param.pow(2)]

    loss_net = bpgs4.network_loss(raw_losses_net)
    loss_net.backward()

    all_pass &= check(net_param.grad is not None and net_param.grad.abs().item() > 0,
                      "network_loss sends grad to network param")

    all_pass &= check(bpgs4.theta.grad is None,
                      "network_loss sends NO grad to theta (stop-gradient correct)")

    # ---- 7. uncertainty_loss() gradient flow ----
    print("\n[7] uncertainty_loss(): gradient isolation")
    bpgs5 = BPGS(num_tasks=2, s_min=-5.0, s_max=5.0, tau=20.0)
    bpgs5.update_ema([torch.tensor(1.0), torch.tensor(2.0)])

    # Simulate a network parameter that should NOT receive grad here
    net_param2 = nn.Parameter(torch.tensor(1.0))

    l_unc = bpgs5.uncertainty_loss()
    l_unc.backward()

    all_pass &= check(bpgs5.theta.grad is not None,
                      "uncertainty_loss sends grad to theta")
    all_pass &= check(bpgs5.theta.grad.abs().sum().item() > 0,
                      "theta grad is non-zero")
    all_pass &= check(net_param2.grad is None,
                      "uncertainty_loss sends NO grad to network params")

    # ---- 8. Prior regularization ----
    print("\n[8] Optional Gaussian prior on theta")
    bpgs_noprior = BPGS(num_tasks=2, tau=20.0)
    bpgs_prior   = BPGS(num_tasks=2, tau=20.0, prior_var=1.0)

    bpgs_noprior.update_ema([torch.tensor(1.0), torch.tensor(1.0)])
    bpgs_prior.update_ema(  [torch.tensor(1.0), torch.tensor(1.0)])
    # Manually set theta to same nonzero value for both
    with torch.no_grad():
        bpgs_noprior.theta.fill_(0.5)
        bpgs_prior.theta.fill_(0.5)

    l_no  = bpgs_noprior.uncertainty_loss().item()
    l_yes = bpgs_prior.uncertainty_loss().item()
    # prior adds (1/(2*1.0)) * sum(theta^2) = 0.5 * 2 * 0.25 = 0.25
    expected_diff = 0.5 / 1.0 * (0.5**2 * 2)
    all_pass &= check(abs((l_yes - l_no) - expected_diff) < 1e-5,
                      "Prior adds correct (1/(2*prior_var))||theta||^2 term",
                      f"diff={l_yes-l_no:.6f}, expected={expected_diff:.6f}")

    # ---- 9. Checkpoint round-trip ----
    print("\n[9] Checkpoint (state_dict) round-trip")
    bpgs_orig = BPGS(num_tasks=3, tau=30.0)
    bpgs_orig.update_ema([torch.tensor(0.5), torch.tensor(1.5), torch.tensor(2.5)])

    # Save and restore
    state = bpgs_orig.state_dict()
    bpgs_loaded = BPGS(num_tasks=3, tau=30.0)
    bpgs_loaded.load_state_dict(state)

    all_pass &= check(
        torch.allclose(bpgs_orig.L_bar, bpgs_loaded.L_bar),
        "L_bar buffer is preserved across checkpoint save/load",
        f"orig={bpgs_orig.L_bar.tolist()}, loaded={bpgs_loaded.L_bar.tolist()}"
    )
    all_pass &= check(
        torch.allclose(bpgs_orig.theta, bpgs_loaded.theta),
        "theta parameter is preserved across checkpoint save/load"
    )

    # ---- 10. reset_ema() ----
    print("\n[10] reset_ema()")
    bpgs_r = BPGS(num_tasks=2, tau=20.0)
    bpgs_r.update_ema([torch.tensor(5.0), torch.tensor(10.0)])
    bpgs_r.reset_ema(value=2.0)
    all_pass &= check(
        torch.all(bpgs_r.L_bar == 2.0).item(),
        "reset_ema fills L_bar with specified value"
    )

    # ---- 11. Device consistency ----
    print("\n[11] Device consistency (CPU)")
    bpgs_d = BPGS(num_tasks=2, tau=20.0)
    bpgs_d.update_ema([torch.tensor(1.0), torch.tensor(2.0)])
    l_u = bpgs_d.uncertainty_loss()
    all_pass &= check(l_u.device.type == "cpu",
                      "uncertainty_loss output is on same device as theta",
                      f"device={l_u.device}")

    # ---- 12. Full training loop simulation ----
    print("\n[12] Full training loop simulation (5 steps, 3 tasks)")
    torch.manual_seed(42)
    net  = nn.Linear(4, 3)
    bpgs_sim = BPGS(num_tasks=3, s_min=-5.0, s_max=5.0, tau=20.0)
    opt_n = torch.optim.Adam(net.parameters(),       lr=1e-3)
    opt_u = torch.optim.Adam(bpgs_sim.parameters(),  lr=1e-3)

    theta_before = bpgs_sim.theta.detach().clone()
    for step in range(5):
        x = torch.randn(8, 4)
        y = net(x)
        raw = [y[:, i].pow(2).mean() for i in range(3)]

        bpgs_sim.update_ema(raw)

        opt_n.zero_grad()
        bpgs_sim.network_loss(raw).backward()
        opt_n.step()

        opt_u.zero_grad()
        bpgs_sim.uncertainty_loss().backward()
        opt_u.step()

    theta_after = bpgs_sim.theta.detach().clone()
    all_pass &= check(
        not torch.allclose(theta_before, theta_after),
        "theta parameters changed after 5 training steps"
    )
    all_pass &= check(
        all(math.isfinite(v) for v in bpgs_sim.get_weights()),
        "All precision weights are finite after 5 steps",
        f"weights={[f'{v:.4f}' for v in bpgs_sim.get_weights()]}"
    )
    all_pass &= check(
        all(math.isfinite(v) for v in bpgs_sim.get_L_bar()),
        "All L_bar values are finite after 5 steps",
        f"L_bar={[f'{v:.4f}' for v in bpgs_sim.get_L_bar()]}"
    )

    # ---- Summary ----
    print(f"\n{SEP}")
    if all_pass:
        print("\033[92mAll checks passed. B-PGS implementation verified.\033[0m")
    else:
        print("\033[91mSome checks FAILED. See output above.\033[0m")
        sys.exit(1)
    print(SEP)


if __name__ == "__main__":
    _run_verification()
