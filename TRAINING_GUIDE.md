# SPECTRA: Mission Training Guide (SOP)
---------------------------------------
*Standard Operating Procedure for Multi-Task Learning Benchmarks*

This guide provides the definitive steps to execute, monitor, and compare the six core MTL methods within the SPECTRA framework.

## 1. Pre-Flight Checklist
Before launching any mission, ensure your environment is synchronized:
- **Workspace**: `c:\Users\Hellx\Documents\Programming\python\Project\iron\bc\SPECTRA`
- **Verification**: Run `python scripts/debug_config.py` to ensure Hydra merges are conflict-free.
- **Telemetry**: Ensure `GradientHealthCallback` is set to `check_interval=1` (standard in `runner.py`) for real-time progress bar visibility.

## 2. Mission Command Reference
Use the following commands to launch standardized 80-epoch synthetic benchmarks or full clinical missions.

### 🏆 Choice A: B-PGS (Recommended)
*Best for: Imbalanced tasks (MSE vs BCE) and high numerical stability.*
```bash
python scripts/train.py dataset=synthetic method=bpgs train.epochs=80 run_name=bench_bpgs
```

### 🥈 Choice B: PCGrad
*Best for: Conflicting gradients and hard multi-task trade-offs.*
```bash
python scripts/train.py dataset=synthetic method=pcgrad train.epochs=80 run_name=bench_pcgrad
```

### 🥉 Choice C: Kendall (Uncertainty)
*Best for: Fast convergence on clean data.*
```bash
python scripts/train.py dataset=synthetic method=kendall train.epochs=80 run_name=bench_kendall
```

### Other Methods:
- **NTK-MTL**: `method=ntkmtl`
- **UWSO**: `method=uwso`
- **Static**: `method=static`

## 3. Real-Time Telemetry (What to Watch)
Monitor the `SOTAProgressBar` for the following health indicators:

| Metric | Target | Warning Range | Action |
| :--- | :--- | :--- | :--- |
| **GN** (Grad Norm) | $0.1 - 10$ | $> 50$ | Check for exploding gradients. |
| **Ratio** (Up/Wt) | $\approx 0.001$ | $> 0.1$ | Learning rate too high. |
| **C** (Conflicts) | $> 0$ | $0$ (for PCGrad) | Tasks are perfectly aligned (rare). |
| **vL** (Val Loss) | Decreasing | Increasing | Overfitting or Manifold Collapse. |

## 4. Benchmarking Protocol (Step-by-Step)
To produce a publication-grade report, follow this sequence:

1.  **Stage 1: Validations (10 Epochs)** - Run each method on synthetic data to ensure no NaNs.
2.  **Stage 2: Benchmarking (80-500 Epochs)** - Run B-PGS, Kendall, and PCGrad side-by-side on synthetic data.
3.  **Stage 3: Clinical Benchmarking** - Select the top 2 performers and launch on the Sepsis dataset:
    ```bash
    python scripts/train.py dataset=clinical method=bpgs train.epochs=10 run_name=clinical_bpgs
    ```
4.  **Stage 4: Comparison** - Collect `val/total_loss` and `val/outcome_AUC` from the logs for the final report.

## 5. Troubleshooting
- **InstantiationException**: Check `configs/dataset/*.yaml`. Ensure `_target_` uses the absolute path (e.g., `spectra.modules.clinical.ClinicalSPECTRAModule`).
- **ValueError (Shape Mismatch)**: This usually happens in the `heads`. ensure that `pred` and `target` dimensions are aligned using the `squeeze(-1)` logic established in `clinical.py`.
- **Vanishing Gradients**: If `Ratio < 1e-4`, check if the task weights have collapsed to 0 (common in primitive Kendall implementations, solved in SPECTRA's B-PGS).

---
**Mission Status: GO FOR TRAINING.**
