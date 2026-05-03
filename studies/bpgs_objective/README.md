# BPGS Objective Study

This directory isolates the publication-facing study for the claim:

> BPGS is a bounded, batch-adaptive, split uncertainty-weighting method whose
> value is robust, consistently competitive behavior under heterogeneous task
> scales and uncertainty dynamics, not universal SOTA performance.

The study is intentionally separated from the generic experiment runners.

## Study Layout

- `definitions.py`
  - single source of truth for executable studies, variants, seeds, and outputs
- `nyuv2_subsets.py`
  - deterministic subset generation for NYUv2 pilot studies
- `run.py`
  - executes one isolated study into `outputs/studies/bpgs_objective/...`
- `aggregate.py`
  - reads completed runs for one study and emits tidy/summarized reports

## Executable Studies

- `01_nyuv2_ablation`
  - fixed NYUv2 subset pilot for ablations
- `02_synthetic_scale_stress`
  - controlled scale-mismatch synthetic stress benchmark
- `03_yeast_regime_check`
  - cross-regime check on Yeast
- `04_qm9_regime_check`
  - cross-regime check on QM9
- `05_nyuv2_full_final`
  - final full NYUv2 seeded comparison

## Notes

- The original single-seed NYUv2 run is treated as prior evidence, not as an
  executable study here.
- This study path reuses the core SPECTRA training/empirical components but
  keeps orchestration, subsets, and aggregation isolated.
