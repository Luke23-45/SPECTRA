# B-PGS: Minimal ML Translation

## 1. Purpose of the translation layer

This document translates the canonical B-PGS definition in [01_bpgs_formal_definition.md](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/SPECTRA/docs/math_definition/final/01_bpgs_formal_definition.md) into the minimal empirical implementation needed for fair comparison.

The goal of this layer is not to maximize practical robustness. The goal is to instantiate the pure method as directly as possible so that experiments test the claimed idea rather than a collection of downstream engineering refinements.

## 2. Symbol-to-code mapping

Use the following symbols consistently in code and manuscript:

| Symbol | Meaning | Minimal implementation object |
|---|---|---|
| $w$ | model parameters | the neural network weights |
| $\theta_i$ | uncertainty coordinate for task $i$ | one scalar learnable parameter per task |
| $s_i(\theta_i)$ | bounded log-variance | `a_i + (b_i - a_i) * sigmoid(theta_i)` |
| $\omega_i(\theta_i)$ | precision | `exp(-s_i(theta_i))` |
| $\ell_i(w)$ | task loss | current batch loss for task $i$ |
| $m_i$ | surrogate task-difficulty statistic | default: `m_i = ell_i(w)` in the pure implementation |
| $J_{\mathrm{net}}$ | network objective | detached precision-weighted loss sum |
| $J_{\mathrm{unc}}$ | uncertainty objective | surrogate-driven uncertainty loss |

The minimal implementation should have:

- one scalar `theta_i` per task,
- fixed task-wise bounds `a_i`, `b_i`,
- one forward computation of task losses,
- one network update pass,
- one uncertainty update pass.

## 3. Minimal training algorithm

The primary implementation for experiments should follow the pure method directly.

### Step 1. Compute task losses

For each batch, compute
$$
L_i = \ell_i(w), \qquad i=1,\dots,T.
$$

These are the current batch losses. In the primary theory-faithful implementation, they are also the surrogate statistics:
$$
m_i = L_i.
$$
No EMA or running average is used in the main comparison implementation. The same batch-level surrogate values are then treated as fixed inputs during the uncertainty update.

### Step 2. Compute bounded log-variance and precision

For each task,
$$
s_i = a_i + (b_i-a_i)\sigma(\theta_i),
\qquad
\omega_i = e^{-s_i}.
$$

### Step 3. Network update

Form
$$
J_{\mathrm{net}}(w \mid \theta)
=
\sum_{i=1}^{T}
\frac{1}{2}\,\operatorname{sg}[\omega_i]\,L_i.
$$

Backpropagate this objective only through the network parameters and update only $w$.

### Step 4. Uncertainty update

Form
$$
J_{\mathrm{unc}}(\theta \mid m)
=
\sum_{i=1}^{T}
\left[
\frac{1}{2}\,\omega_i\,m_i
+
\frac{1}{2}\,s_i
\right].
$$

Backpropagate this objective only through the uncertainty parameters and update only $\theta$.
In the minimal implementation, the same pre-update batch values computed in Step 1 are reused as the surrogate inputs in Step 4. In practice, this means `m_i` should be detached or otherwise treated as constant with respect to the uncertainty update.

### Step 5. Optimizer structure

The cleanest realization uses either:

- two optimizers, one for $w$ and one for $\theta$, or
- one optimizer with two explicit zero-grad / backward / step phases that preserve the split update semantics.

The paper should present the two-optimizer version because it makes the separation easiest to verify.

## 4. Fair comparison rules

The main experimental comparison should test the pure idea under matched conditions.

### Required fairness rule

The primary head-to-head evaluation must compare:

- pure B-PGS,
- pure Kendall-style uncertainty weighting,
- and any additional uncertainty-weighting baselines,

under the same backbone, data splits, optimizer family, learning-rate protocol, batch schedule, and number of training steps.

### What must not happen

Do not give B-PGS engineering advantages in the main comparison that the baselines do not receive. In particular, the main table should not allow:

- B-PGS-only EMA smoothing,
- B-PGS-only NaN filters,
- B-PGS-only mixed-precision guards,
- B-PGS-only custom calibration tricks,
- B-PGS-only distributed synchronization logic that changes the effective method.

If such techniques are studied, they should appear in ablations or robustness appendices, not in the primary fairness-critical claim.

## 5. Optional engineering extensions kept out of the core paper claim

The production implementation may include additional mechanisms for robustness or convenience. These are legitimate engineering choices, but they must be kept separate from the core paper claim.

Examples include:

- EMA-smoothed uncertainty targets,
- strictly positive surrogates such as `softplus` or `R_eps`,
- NaN-gates or finite-value filters,
- fp16/fp32 casting safeguards,
- DDP synchronization logic,
- warm-start and auto-calibration heuristics.

These extensions should be discussed only as:

- practical deployment refinements,
- robustness enhancements,
- or ablation variants.

They should not redefine the canonical method and should not be the reason the primary B-PGS line outperforms baselines in the main paper table.

## 6. Practical note on the existing repo implementation

The current implementation in [spectra/core/bpgs.py](/C:/Users/Hellx/Documents/Programming/python/Project/iron/bc/SPECTRA/spectra/core/bpgs.py) should be treated as a practical descendant implementation, not as the canonical mathematical source.

That module can still be valuable for:

- secondary robustness experiments,
- implementation ablations,
- and future production use.

But the first empirical evaluation for the paper should be the minimal translation described here.
