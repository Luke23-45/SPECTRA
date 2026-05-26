# BPGS Formalization

This note is an implementation-aligned description of the BPGS objective used in this repository. It is not a proof document and it does not claim properties that are not directly encoded in the code.

## Code References

Primary implementation files:

- `spectra/core/bpgs.py`
- `spectra/engine/optimizers/bpgs.py`

Canonical training-facing configuration:

- `configs/method/bpgs.yaml`

Canonical empirical benchmark architecture defaults:

- `configs/empirical/bpgs/default.yaml`

## Canonical Modes Used in This Repository

The default BPGS settings used by the standard training config path are:

- `s_mode = batch_aware`
- `init_mode = auto_calibrate`

Those defaults are defined in `configs/method/bpgs.yaml`.

## Notation

Let `T` be the number of tasks and let `L_i(w) >= 0` denote the task loss for task `i`.

Define the detached log-loss statistics:

$$
\widetilde{L}_i = \max(L_i, \varepsilon_{\log}),
\qquad
\mu(L) = \frac{1}{T}\sum_{i=1}^{T}\log \widetilde{L}_i
$$

$$
\bar{\sigma}(L) =
\max\left(
\sqrt{\frac{1}{T}\sum_{i=1}^{T}\left(\log \widetilde{L}_i - \mu(L)\right)^2},
\varepsilon_{\mathrm{std}}
\right)
$$

Each task has a learnable coordinate `theta_i in R`, and the latent radius is:

$$
\tau_T = \sqrt{T - 1} + 0.1
$$

## Symbol-to-Code Mapping

| Symbol | Meaning | Code |
|--------|---------|------|
| `T` | Number of tasks | `num_tasks` |
| `s_min`, `s_max` | Stateless chart bounds | `self.s_min`, `self.s_max` |
| `eps_log` | Log clamp | `self.eps_clip` |
| `eps_std` | Std floor | `1e-4` in `_update_batch_stats()` |
| `gamma_theta` | Gradient scale applied to `theta` | `self.theta_grad_scale` via `GradScale` |
| `tau_T` | Latent radius | `self.topological_limit` |
| `theta_i` | Uncertainty coordinate | `self.theta[i]` |
| `mu(L)` | Mean detached log-loss | `log_losses.mean()` |
| `bar_sigma(L)` | Clamped detached log-loss std | `log_losses.std(unbiased=False).clamp(min=1e-4)` |
| `omega_i` | Raw precision | `torch.exp(-s)` |
| `alpha_i` | Normalized network weight | `weights / weights.sum()` |

## Batch-Aware Chart

The batch-aware latent coordinate is:

$$
z_i(\theta_i) = \tau_T \left(2 \sigma(\theta_i) - 1\right)
$$

The batch-aware log-variance is:

$$
s_i(\theta_i; L) = \mu(L) + \bar{\sigma}(L) z_i(\theta_i)
$$

The corresponding precisions and normalized network weights are:

$$
\omega_i = e^{-s_i},
\qquad
\alpha_i = \frac{\omega_i}{\sum_{j=1}^{T} \omega_j}
$$

## Stateless Ablation Chart

For the stateless ablation path, the implementation uses:

$$
s_i^{\mathrm{stateless}}(\theta_i) =
s_{\min} + (s_{\max} - s_{\min}) \sigma(\theta_i)
$$

This path is used by the ablation study when the corresponding Hydra overrides are applied through the study runner.

## Objectives

The network update uses detached normalized weights:

$$
J_{\mathrm{net}}(w \mid \theta, L) =
\sum_{i=1}^{T} \operatorname{sg}[\alpha_i] L_i
$$

The uncertainty update uses detached losses:

$$
J_{\mathrm{unc}}(\theta \mid L) =
\sum_{i=1}^{T}
\left[
\frac{1}{2}\omega_i \operatorname{sg}[L_i] + \frac{1}{2}s_i
\right]
$$

The split optimizer in `spectra/engine/optimizers/bpgs.py` applies these objectives in separate steps:

- one optimizer for network parameters
- one optimizer for the BPGS uncertainty parameters

## Auto-Calibration

On the first observed batch, the auto-calibration path computes:

$$
z_i^{\mathrm{raw}} =
\frac{\log \widetilde{L}_i - \mu(L)}{\bar{\sigma}(L)},
\qquad
z_i^{\mathrm{clip}} =
\operatorname{clamp}(z_i^{\mathrm{raw}}, -\tau_T, \tau_T)
$$

$$
p_i = \frac{z_i^{\mathrm{clip}} + \tau_T}{2\tau_T},
\qquad
\widetilde{p}_i = \operatorname{clamp}(p_i, 10^{-4}, 1 - 10^{-4})
$$

$$
\theta_i \leftarrow \log \left(\frac{\widetilde{p}_i}{1 - \widetilde{p}_i}\right)
$$

This is executed once and then guarded by `self._calibrated`.

## Directly Checkable Implementation Properties

For finite inputs, the implementation directly implies:

- `omega_i > 0`
- `alpha_i > 0`
- `sum_i alpha_i = 1`
- `s_i(theta_i; L)` is bounded for fixed batch statistics because `z_i` is bounded in `[-tau_T, tau_T]`
- the network objective does not backpropagate through `alpha_i`
- the uncertainty objective does not backpropagate through `L_i`

## Pseudocode

```text
Input: network parameters w, uncertainty coordinates theta, batch losses L

1. If init_mode == auto_calibrate and not calibrated:
       theta <- auto_calibrate(L)
2. Compute detached log-loss statistics mu and sigma from L
3. Map theta into bounded latent coordinates z
4. Construct s from either:
       stateless chart, or
       batch-aware chart mu + sigma * z
5. Compute omega = exp(-s)
6. Normalize weights alpha = omega / sum(omega)
7. Update network parameters with sum_i stopgrad(alpha_i) * L_i
8. Update uncertainty parameters with sum_i [0.5 * omega_i * stopgrad(L_i) + 0.5 * s_i]
```

## Claims Not Made

This document does not claim:

- a full Bayesian derivation
- a convergence proof
- superiority over all baselines on all benchmarks

It is an implementation-aligned description of what this repository actually optimizes.
