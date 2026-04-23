# SPECTRA Benchmarking Scripts

This directory contains the automated execution pipelines for the SPECTRA Multi-Task Learning framework.

## 1. Synthetic Gauntlet
The `run_synthetic` scripts execute an unbroken convergence test across the active paper-track method set:
1. `static` (Equal Weighting)
2. `kendall` (Uncertainty Weighting)
3. `uwso` (Uncertainty Weighting + Scalar Optimization)
4. `gradnorm_proxy` (Gradient-Norm Proxy Baseline)
5. `pcgrad` (Projecting Conflicting Gradients)
6. `bpgs` (Bounded Precision Gradient Splitting)

`bpgs_alb` is retained only as an archived experimental composite and is not part of the primary paper-track benchmark set.

### How to Run (Windows)
**Option A: PowerShell (Recommended)**
Open PowerShell in the root `SPECTRA` directory and run:
`.\scripts\benchmarks\run_synthetic.ps1`

**Option B: Command Prompt**
Open Command Prompt in the root `SPECTRA` directory and run:
`scripts\benchmarks\run_synthetic.bat`

### Outputs
- **Logs:** Highly descriptive CSV logs will be continuously written to: `outputs/{method}_synthetic_s42/csv_logs/{timestamp}/metrics.csv`.
- **Checkpoints:** The top 3 performing binary weights will be safely preserved in `outputs/`.
- **Interruption Safety:** If any script fails, the runner will immediately halt to prevent cascading failures.
