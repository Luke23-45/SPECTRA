# SPECTRA

SPECTRA is a multi-task learning research framework centered on Bounded Precision-Geometry Scaling (BPGS) and a small set of baseline task-weighting methods. The repository includes standard training entry points, study orchestration for the BPGS paper, dataset preparation utilities, and a separate analysis pipeline for figures, tables, and statistics.

## Project Structure

```text
SPECTRA/
|-- spectra/                     # Core Python package
|   |-- architectures/          # Model builders
|   |-- backbones/              # SegNet and shared-trunk backbones
|   |-- baselines/              # Kendall, PCGrad, UWSO, Static, GradNorm proxy
|   |-- core/                   # BPGS, gated fusion, lateral bypass
|   |-- data/                   # Dataset loaders and ingest modules
|   |-- empirical/              # Synthetic benchmark framework
|   |-- engine/                 # Losses, callbacks, schedulers, optimizers
|   |-- evaluation/             # Metrics
|   |-- heads/                  # Task-specific prediction heads
|   |-- modules/                # Lightning modules for vision, synthetic, clinical
|   |-- train/                  # Training orchestration and artifact logging
|   |-- utils/                  # Config, seeding, W&B, progress helpers
|   `-- visualization/          # Figure-generation helpers
|-- configs/                    # Hydra configuration
|   |-- config.yaml             # Default training config
|   |-- dataset/                # Dataset configs
|   |-- empirical/             # Synthetic benchmark configs
|   |-- method/                 # Method configs
|   |-- preset/                 # Publication-ready training presets
|   `-- sweeps/                 # Sweep configs
|-- experiments/
|   |-- bpgs_study/             # Study execution
|   `-- bpgs_analysis/          # Post-run analysis pipeline
|-- scripts/
|   |-- run_training.py         # Hydra training entry point
|   |-- run_empirical_experiment.py
|   |-- data/                   # Dataset preparation scripts
|   `-- execution/              # Reproduction utilities
|-- docs/                       # Project documentation
`-- working/                    # Paper drafts and notes
```

## Quickstart

```bash
pip install -e .

# Synthetic training with BPGS
python scripts/run_training.py method=bpgs seed=42

# NYUv2 training
python scripts/run_training.py preset=nyuv2_bpgs seed=42

# List registered studies
python experiments/bpgs_study/run.py --list

# Preview a study plan without executing it
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --dry-run
```

## Methods

| Method | Config | Description |
|--------|--------|-------------|
| BPGS | `configs/method/bpgs.yaml` | Bounded Precision-Geometry Scaling |
| Kendall | `configs/method/kendall.yaml` | Homoscedastic uncertainty weighting |
| PCGrad | `configs/method/pcgrad.yaml` | Projecting Conflicting Gradients |
| UWSO | `configs/method/uwso.yaml` | Uncertainty-Weighted Soft Optimization |
| Static | `configs/method/static.yaml` | Fixed equal weighting |
| GradNorm Proxy | `configs/method/gradnorm_proxy.yaml` | GradNorm-style proxy baseline |

## Datasets

| Dataset | Type | Tasks | Config |
|---------|------|-------|--------|
| Synthetic | Generated benchmark | 7 default tasks | `configs/dataset/synthetic.yaml` |
| NYUv2 | Dense prediction | Segmentation, depth, normals | `configs/dataset/nyuv2.yaml` |
| Yeast | Multi-label classification | 14 binary labels | `configs/dataset/yeast.yaml` |
| RF1 | Multi-target regression | 8 flow-site targets | `configs/dataset/rf1.yaml` |
| QM9 | Molecular regression | 12 properties | `configs/dataset/qm9.yaml` |
| Clinical | Clinical prediction | Outcome and phase | `configs/dataset/clinical.yaml` |

## Documentation

- [Getting Started](docs/getting_started.md)
- [Studies](docs/studies.md)
- [Datasets](docs/datasets.md)
- [Analysis](docs/analysis.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [BPGS Formalization](docs/bpgs_formalization.md)

## Requirements

- Python 3.9 or newer
- PyTorch 2.1 or newer
- See `pyproject.toml` for the full dependency list
