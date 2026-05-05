# Empirical Experiments

This package contains the publication-facing synthetic benchmark framework for evaluating task-weighting methods under controlled correlation, imbalance, and conflict stressors.

Primary entrypoints:

- `python -m spectra.empirical.experiments.exp_01_correlation_sweep`
- `python -m spectra.empirical.experiments.exp_02_conflict_stability`
- `python -m spectra.empirical.experiments.exp_03_imbalance_robustness`
- `python -m spectra.empirical.experiments.exp_04_control_benign`
- `python -m spectra.empirical.experiments.exp_05_multiclass_behavior`
- `python -m spectra.empirical.experiments.exp_06_orchestrator`

Results are written under `outputs/empirical/<family>/<method>/seed_<seed>/`.
