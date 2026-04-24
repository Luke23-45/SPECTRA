# BPGS Math Breakdown (NYUv2-focused)

## 1) Canonical split objective
Given task losses \(L_i\) and bounded uncertainty chart
\[
s_i = s_{min} + (s_{max}-s_{min})\sigma(\theta_i), \quad \omega_i = e^{-s_i},
\]
SPECTRA optimizes
\[
J_{net} = \sum_i \tfrac{1}{2}\,\mathrm{stopgrad}(\omega_i)\,L_i,
\]
\[
J_{unc} = \sum_i \left[\tfrac{1}{2}\,\omega_i\,\tilde L_i + \tfrac{1}{2}s_i\right].
\]

## 2) Why raw-loss BPGS can underperform
If \(\tilde L_i=L_i\) (raw detached losses), stationarity of the uncertainty flow gives
\[
\partial J_{unc}/\partial s_i = -\tfrac{1}{2}e^{-s_i}L_i + \tfrac{1}{2}=0
\Rightarrow s_i^* = \log L_i.
\]
So weights become \(\omega_i^*=1/L_i\). This is mathematically valid, but strongly tied to absolute units of each task loss (segmentation CE vs depth L1 vs normals cosine).

## 3) Fundamental fix used in this patch
We keep the same BPGS geometry and split optimization, but normalize detached losses in the uncertainty branch:
\[
\tilde L_i = \frac{L_i}{\mathrm{Norm}(L)},
\]
where `Norm` is configurable (`max`, `mean`, `geo`, `raw`) and defaults to `max` for the NYUv2 investigation.

This preserves relative difficulty while removing global scale coupling, i.e., if all losses are multiplied by constant \(c\), \(\tilde L_i\) remains unchanged.

## 4) Practical implication
- Network branch remains unchanged: still optimizes weighted raw losses.
- Uncertainty branch becomes scale-invariant (when enabled), reducing pathological shifts from heterogeneous task magnitudes.
- No EMA, no auxiliary training tricks, no changes to backbone/head optimization.


## 5) Fairness note for experiments
- `bpgs` should remain the raw baseline.
- The normalized uncertainty update should be evaluated as a separate variant (`bpgs_scaleinv`).
- Comparison is fair only when compute budget, data, seeds, and evaluation are identical.
