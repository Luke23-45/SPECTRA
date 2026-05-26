# Getting Started

This document describes the repository's executable entrypoints and the minimum steps needed to run them correctly.

## Prerequisites

- Python 3.9 or newer
- PyTorch 2.1 or newer
- CUDA is optional for the repository as a whole, but several NYUv2 and larger study paths are configured assuming GPU availability

The declared package dependencies live in `pyproject.toml`.

## Installation

```bash
git clone <repo-url> SPECTRA
cd SPECTRA
pip install -e .
```

Optional development dependencies:

```bash
pip install -e ".[dev]"
```

## Runtime Environment

The training entrypoint `scripts/run_training.py` sets:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

if the variable is not already defined. The repository also includes `python-dotenv` as a dependency, so a project-root `.env` file can be used when local environment variables are needed.

## Verify Installation

```bash
python -c "import spectra; print(spectra.__version__)"
```

The version string is defined in `spectra/__init__.py`.

## Primary Entry Points

### Standard training

The standard training entrypoint is:

```bash
python scripts/run_training.py
```

This composes the Hydra config rooted at `configs/` and delegates execution to `spectra.train.runner.execute_training_mission(...)`.

Useful first commands:

```bash
python scripts/run_training.py
python scripts/run_training.py method=bpgs seed=42
python scripts/run_training.py dataset=yeast method=kendall seed=42
python scripts/run_training.py preset=nyuv2_bpgs seed=42
```

### Empirical synthetic benchmark

The controlled empirical benchmark entrypoint is:

```bash
python scripts/run_empirical_experiment.py experiment=exp_01_correlation_sweep
```

This composes the Hydra config rooted at `configs/empirical/`, validates the experiment config, initializes the empirical run context, and dispatches to an experiment registered in `spectra/empirical/experiments/__init__.py`.

### Paper study runner

The study runner used for paper-oriented execution plans is:

```bash
python experiments/bpgs_study/run.py --list
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --dry-run
```

This runner does not define experiments itself. It loads study specs from `experiments/bpgs_study/*/configs/` and generates the corresponding training or empirical commands.

### Analysis pipeline

The analysis pipeline is run from `experiments/bpgs_analysis/` because the modules import `analysis.*`:

```bash
cd experiments/bpgs_analysis
python -m analysis.scripts.generate_all
```

## Dataset Materialization Commands

Some datasets require an explicit preparation step before training:

```bash
python scripts/data/nyuv2_generate.py
python scripts/data/yeast_generate.py
python scripts/data/rf1_generate.py
python scripts/data/qm9_generate.py
```

The preparation code is defined in:

- `spectra/data/nyuv2/nyuv2_lmdb_sota.py`
- `spectra/data/yeast/ingest.py`
- `spectra/data/rf1/ingest.py`
- `spectra/data/qm9/ingest.py`

Clinical data is resolved lazily by `ensure_data_ready(...)` in `spectra/data/clinical/dataset.py`.

## Output Conventions

Standard training runs write under the resolved `output_dir` from the composed Hydra config. By default that is:

```text
outputs/<run_name>/
```

Empirical runs write under the empirical output root configured in `configs/empirical/output/default.yaml`.

Study-run outputs are organized under:

```text
outputs/experiments/bpgs_study/
```

The analysis scripts default to writing under:

- `analysis/outputs/paper_final` for `generate_all.py`
- `analysis/outputs` for the figure, table, and stats scripts when run individually

## Before Running Camera-Ready Paths

For paper-facing reproduction, verify the following first:

- the required dataset is already materialized locally or can be downloaded by the corresponding pipeline
- the chosen method and dataset compose successfully through Hydra
- NYUv2 runs have adequate GPU memory and prepared LMDB files
- study dry-runs produce the command sequence you expect

Useful checks:

```bash
python scripts/run_training.py --cfg job dataset=rf1 method=bpgs
python scripts/run_empirical_experiment.py --cfg job experiment=exp_03_imbalance_robustness
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --dry-run
```
