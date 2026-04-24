# BPGS Objective Breakdown and Conflict-Reduction Variant

## 1) Why the canonical uncertainty loss can underperform

Canonical uncertainty flow (per task):

\[
J_i(s_i; L_i) = \frac{1}{2} e^{-s_i} L_i + \frac{1}{2} s_i
\]

Gradient wrt \(s_i\):

\[
\frac{\partial J_i}{\partial s_i} = \frac{1}{2}\left(1 - e^{-s_i} L_i\right)
\]

Stationary condition:

\[
e^{-s_i}L_i = 1 \Rightarrow s_i^\star = \log L_i
\]

So for fixed \(L_i\), the math is well-behaved. In practice, \(L_i\) keeps moving during training, and the optimizer sees a drifting objective with two competing terms. This can lead to oscillation/lag in finite-step training.

## 2) Direct target-matching reformulation

Since the analytic target is \(s_i^\star = \log L_i\), we can optimize:

\[
J_i^{\text{log-match}} = \frac{1}{2}\left(s_i - \log(L_i + \varepsilon)\right)^2
\]

or robustly:

\[
J_i^{\text{huber-log}} = \operatorname{Huber}\left(s_i,\,\log(L_i + \varepsilon)\right)
\]

This preserves the same bounded chart for \(s_i\), but removes the internal tug-of-war inside the uncertainty objective.

## 3) What changed in code

`BPGS` now supports:
- `unc_mode: kendall` (original)
- `unc_mode: log_mse` (direct log-target matching)
- `unc_mode: huber_log` (robust target matching)

Network loss is unchanged and still uses detached precision weighting.

## 4) Recommendation for fair comparison

For the main fairness table:
- keep `unc_mode: kendall` as canonical BPGS,
- report `log_mse` and `huber_log` as explicit math ablations,
- use identical backbone/splits/optimizer/scheduler/steps across methods.
