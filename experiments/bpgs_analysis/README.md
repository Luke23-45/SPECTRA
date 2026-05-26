# BPGS Experiment

This directory contains all BPGS (Batch-Projected Gradient Scaling) experiment data and analysis for the SPECTRA project. It is organized into two main subdirectories:

- **`data/`** — Raw experiment outputs organized by category: **ablation**, **full-data regime checks**, **NYUv2 benchmark**, and **stress tests**.
- **`analysis/`** — Publication-quality analysis pipeline (tables, figures, statistics) that reads from `data/`.

Each experiment run follows a consistent output structure:

```
<method>/<seed_N>/
├── config.yaml          # Full Hydra config snapshot for the run
├── metadata.json        # Experiment, system, git, dataset, and training metadata
├── run_summary.json     # Fit timing, stopped epoch, selection metric, checkpoint registry
├── run_training.log     # Verbose training log (where available)
└── csv_logs/
    └── <run_id>/
        └── metrics.csv  # Per-epoch metric traces (train + validation)
```

Stress-test runs additionally contain:

```
├── metrics/
│   ├── results.csv       # Per-method raw results table
│   ├── results.json      # Same data in JSON
│   ├── results.jsonl     # Line-delimited JSON variant
│   ├── summary.csv       # Aggregated summary (mean, std, count, stderr)
│   └── summary.json      # Aggregated summary in JSON
├── figures/
│   ├── <method>_seed_N_trajectory.png   # Loss/metric trajectory plot
│   ├── macro_score_heatmap.png          # Cross-method score heatmap
│   └── macro_score_boxplot.png          # Score distribution boxplot
├── logs/
│   └── run.log           # Experiment runner log
└── artifacts/
    └── run_metadata.json # Run-level metadata (experiment name, timestamp, git info)
```

---

## Directory Layout

### `data/abalation/`

Ablation study outputs testing BPGS variant configurations on the **NYUv2** dataset. Each variant is run with 3 seeds (42, 43, 44).

- **`console_logs/`** — Captured console output (Markdown) from each ablation run:
  - `bpgs_batch_aware_fixed.md` — BPGS with batch-aware fixed scaling
  - `bpgs_canonical.md` — Standard/canonical BPGS configuration
  - `bpgs_stateless_auto.md` — Stateless auto-scaling BPGS variant
  - `bpgs_stateless_fixed.md` — Stateless fixed-scaling BPGS variant
  - `kendall.md` — Kendall uncertainty weighting baseline

- **`final_outputs/`** — Structured run outputs per method variant. Each method directory contains:
  - Per-seed run directories (`bpgs_nyuv2_s42`, `bpgs_nyuv2_s43`, `bpgs_nyuv2_s44` for BPGS variants; `kendall_nyuv2_s42` etc. for Kendall) with `config.yaml`, `metadata.json`, `run_summary.json`, and `csv_logs/<run_id>/metrics.csv`
  - `studies/` — Mirrored output path with `run_training.log` files under the study hierarchy (`bpgs_study/01_nyuv2_ablation/<method>/seed_N/`)

  Methods present:
  - `bpgs_batch_aware_fixed` — Batch-aware BPGS with fixed scaling
  - `bpgs_canonical` — Canonical BPGS
  - `bpgs_stateless_auto` — Stateless BPGS with automatic scaling
  - `bpgs_stateless_fixed` — Stateless BPGS with fixed scaling
  - `kendall` — Kendall uncertainty weighting baseline

- **`zip_versions/`** — Timestamped ZIP archives of ablation outputs (5 files, dated 2026-05-05)

---

### `data/full_data/`

Full-scale experiment outputs on real-world multi-task datasets (Yeast, RF1) comparing multiple methods across seeds.

- **`console_logs/`** — Captured console output (Markdown):
  - `03_yeast_regime_check.md` — Yeast dataset regime check console output
  - `08_rf1_regime_check.md` — RF1 dataset regime check console output

- **`final_results/`** — Structured run outputs organized by experiment and method:
  - `03_yeast_regime_check/` — Yeast multi-label classification (14 binary classification tasks, BCE loss). Methods: `bpgs`, `gradnorm_proxy`, `kendall`, `pcgrad`, `static`, `uwso`. Each method has `seed_42/`, `seed_43/`, `seed_44/` with `config.yaml`, `metadata.json`, `run_summary.json`, `run_training.log`, and `csv_logs/<run_id>/metrics.csv`
  - `08_rf1_regime_check/` — RF1 multi-target regression (8 regression targets). Same method set and seed structure as Yeast.

- **`zip_version/`** — Archived outputs:
  - `spectra_outputs_no_checkpoints.zip` — Full output archive excluding model checkpoints
  - `spectra_outputs_no_checkpoints/` — Extracted content mirror (`content/SPECTRA/outputs/studies/bpgs_study/`)

---

### `data/nyuv2/`

NYUv2 dense prediction benchmark outputs (segmentation, depth, normals) comparing methods across seeds.

- **`final_output_versions/`** — Structured run outputs per method:
  - `bpgs/bpgs/` — BPGS runs (note: double-nested `bpgs/bpgs/` path) with `seed_42/`, `seed_43/`, `seed_44/` (each containing `config.yaml`, `metadata.json`, `run_summary.json`, `run_training.log`, `csv_logs/<run_id>/metrics.csv`)
  - `kendall/` — Kendall uncertainty weighting (`seed_42/`, `seed_43/`, `seed_44/`; each with `config.yaml`, `metadata.json`, `run_summary.json`, `run_training.log`, `csv_logs/`)
  - `static/static/` — Static/equal weighting baseline (note: double-nested `static/static/` path) with `seed_42/`, `seed_43/`, `seed_44/` (same file set as above)
  - `uwso/` — Uncertainty-Weighted Soft Optimization (`seed_42/`, `seed_43/`, `seed_44/`; same file set as above)

- **`zip_versions/`** — Per-method ZIP archives for easy sharing:
  - `bpgs.zip` — BPGS method outputs
  - `kendall.zip` — Kendall method outputs
  - `static.zip` — Static weighting outputs
  - `uwso.zip` — UWSO method outputs

---

### `data/stress/`

Stress-test experiment outputs evaluating method robustness under extreme conditions.

- **`final_outputs/`** — Organized by stress-test experiment:
  - `02_synthetic_scale_stress/` — Synthetic task scaling stress test. Subdirectories `x1/`, `x10/`, `x100/`, `x1000/` represent increasing loss scale factors. Each scale level contains method directories (`bpgs/`, `kendall/`, `uwso/`) with per-seed runs. Also includes `exp_03_imbalance_robustness/latest_run.json` metadata.
  - `06_pure_loss_rescaling/` — Pure loss magnitude rescaling stress test. Same `x1/`–`x1000/` scale structure with `bpgs/`, `kendall/`, `uwso/` methods. Includes `exp_07_pure_loss_rescaling/latest_run.json`.
  - `07_heterogeneous_mixed_stress/` — Heterogeneous mixed-task regime with diverse loss types and metrics. Organized by noise condition:
    - `clean/` — Clean regime (`bpgs/`, `kendall/`, `pcgrad/`, `uwso/`)
    - `noisy/` — Noisy regime (`bpgs/`, `kendall/`, `pcgrad/`, `uwso/`)
    - `conflict/` — Conflict regime (`bpgs/`, `kendall/`, `pcgrad/`, `uwso/`)
    - `exp_08_heterogeneous_regime/latest_run.json` — Run metadata
    - Each method directory contains per-seed runs with `metrics/` (results, summary), `figures/` (trajectory, heatmap, boxplot), `logs/`, and `artifacts/` subdirectories.

- **`zip_version/`** — Archived stress-test outputs:
  - `spectra_outputs (4).zip`

---

## Methods Reference

| Method | Description |
|--------|-------------|
| `bpgs` | Batch-Projected Gradient Scaling (canonical) |
| `bpgs_batch_aware_fixed` | BPGS with batch-aware fixed scaling |
| `bpgs_stateless_auto` | Stateless BPGS with automatic scaling |
| `bpgs_stateless_fixed` | Stateless BPGS with fixed scaling |
| `kendall` | Kendall uncertainty weighting |
| `gradnorm_proxy` | GradNorm proxy gradient normalization |
| `pcgrad` | Projecting Conflicting Gradients |
| `static` | Equal/static task weighting baseline |
| `uwso` | Uncertainty-Weighted Soft Optimization |

## Datasets Reference

| Dataset | Type | Tasks |
|---------|------|-------|
| NYUv2 | Dense prediction | Segmentation (cross-entropy), Depth (masked L1), Normals (cosine) |
| Yeast | Multi-label classification | 14 binary labels (BCE) |
| RF1 | Multi-target regression | 8 flow-site regression targets (MSE) |
