# Configuration

SPECTRA uses Hydra for both standard training runs and the empirical synthetic benchmark suite, but the repository intentionally keeps these as two separate configuration roots because they drive different entrypoints and different schemas.

## Configuration Roots

| Config root | Entry point | Purpose |
|------------|-------------|---------|
| `configs/` | `scripts/run_training.py` | Standard training on the repository's training datasets |
| `configs/empirical/` | `scripts/run_empirical_experiment.py` | Controlled synthetic benchmark experiments for the paper |

The training entrypoint is defined in `scripts/run_training.py`. The empirical entrypoint is defined in `scripts/run_empirical_experiment.py`.

## Training Configuration Layout

```text
configs/
|-- config.yaml
|-- dataset/
|   |-- clinical.yaml
|   |-- nyuv2.yaml
|   |-- qm9.yaml
|   |-- rf1.yaml
|   |-- synthetic.yaml
|   `-- yeast.yaml
|-- method/
|   |-- bpgs.yaml
|   |-- gradnorm_proxy.yaml
|   |-- kendall.yaml
|   |-- pcgrad.yaml
|   |-- static.yaml
|   `-- uwso.yaml
|-- preset/
|   `-- nyuv2_bpgs.yaml
`-- sweeps/
    `-- weight_decay_sweep.yaml
```

### Training composition contract

The base training defaults live in `configs/config.yaml`:

- `dataset: synthetic`
- `method: pcgrad`
- `optional preset: null`

The dataset group defines:

- data identity
- task list
- module target
- dataset-specific model defaults
- dataset-specific training defaults

The method group defines:

- `method_name`
- `method.*` parameters
- any engine override required by that method
- any method-specific training overrides

The preset group exists for named paper-facing combinations that should be launched as a unit. At present the repository includes one training preset:

- `preset=nyuv2_bpgs` in `configs/preset/nyuv2_bpgs.yaml`

## Empirical Configuration Layout

```text
configs/empirical/
|-- config.yaml
|-- analysis/
|   `-- default.yaml
|-- bpgs/
|   `-- default.yaml
|-- experiment/
|   |-- exp_01_correlation_sweep.yaml
|   |-- exp_02_conflict_stability.yaml
|   |-- exp_03_imbalance_robustness.yaml
|   |-- exp_04_control_benign.yaml
|   |-- exp_05_multiclass_behavior.yaml
|   |-- exp_06_orchestrator.yaml
|   |-- exp_07_pure_loss_rescaling.yaml
|   `-- exp_08_heterogeneous_regime.yaml
|-- family/
|   |-- conflict.yaml
|   |-- correlation.yaml
|   |-- heterogeneous.yaml
|   |-- imbalance.yaml
|   `-- rescaling.yaml
|-- logging/
|   `-- default.yaml
|-- methods/
|   |-- boundedness.yaml
|   |-- multiclass.yaml
|   |-- paper_core.yaml
|   |-- stability.yaml
|   `-- stress_full.yaml
|-- output/
|   `-- default.yaml
|-- runtime/
|   `-- default.yaml
`-- seeds/
    `-- default.yaml
```

The empirical base defaults live in `configs/empirical/config.yaml`. They select:

- runtime
- output
- logging
- analysis
- seeds
- methods
- bpgs
- family
- experiment

Each file under `configs/empirical/experiment/` may override:

- `/family`
- `/methods`
- `experiment.*`

The generator schema itself is validated in `spectra/empirical/generators/base.py`, and runtime validation is enforced in `spectra/empirical/utils/config_utils.py`.

## Common Training Overrides

Hydra overrides for standard training:

```bash
python scripts/run_training.py dataset=yeast method=bpgs
python scripts/run_training.py seed=43 train.epochs=100 train.lr=0.001
python scripts/run_training.py preset=nyuv2_bpgs
```

The user-facing invariant is simple:

- `dataset=...` selects a dataset config from `configs/dataset/`
- `method=...` selects a method config from `configs/method/`
- `preset=...` selects a named preset from `configs/preset/`

## Common Empirical Overrides

Hydra overrides for the controlled synthetic benchmark:

```bash
python scripts/run_empirical_experiment.py experiment=exp_01_correlation_sweep
python scripts/run_empirical_experiment.py experiment=exp_03_imbalance_robustness methods.names=[bpgs,uwso]
python scripts/run_empirical_experiment.py experiment=exp_07_pure_loss_rescaling experiment.training_stress.loss_scale_multiplier=100.0
```

The empirical runtime requires the following top-level groups after composition:

- `experiment`
- `family`
- `methods`
- `runtime`
- `output`
- `seeds`

That requirement is enforced in `spectra/empirical/utils/config_utils.py`.

## Study-Scoped Overrides

The paper study runner introduces a second override layer on top of Hydra. The parser for this behavior is defined by `OverrideSpec` in `experiments/bpgs_study/common/specs.py`.

Accepted forms:

- `key=value`
- `method_name:key=value`

Examples:

```bash
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --set bpgs:method.s_mode=stateless
python experiments/bpgs_study/run.py --study 03_yeast_regime_check --set train.batch_size=8
```

The first applies only to variants whose `method` matches `bpgs`. The second applies to every selected variant.

## Key Training Fields

The exact values depend on the selected dataset and method, but these are the main user-facing controls in the training tree:

| Field | Meaning |
|------|---------|
| `seed` | Global run seed |
| `resume_from` | Resume policy or explicit checkpoint path |
| `require_cuda` | Hard-fail if CUDA is unavailable |
| `output_dir` | Stable run artifact root |
| `logging.use_wandb` | Enable or disable Weights & Biases logging |
| `train.epochs` | Maximum epoch count |
| `train.batch_size` | Batch size |
| `train.lr` | Base learning rate |
| `train.precision` | PyTorch Lightning precision mode |
| `train.selection_metric` | Metric used for checkpoint selection |
| `train.selection_mode` | Optimization direction for the selection metric |

The code that consumes these fields is primarily in:

- `spectra/train/runner.py`
- `spectra/train/preflight.py`
- `spectra/train/artifacts.py`
- `spectra/data/datamodule.py`

## Analysis Configuration

The analysis pipeline has its own OmegaConf configuration file:

- `experiments/bpgs_analysis/analysis/configs/base.yaml`

That file defines:

- path aliases to processed study outputs
- seed lists
- method lists
- table formatting options
- figure settings
- export settings

The analysis scripts are plain argparse entrypoints, not Hydra entrypoints. Their behavior is defined in:

- `experiments/bpgs_analysis/analysis/scripts/generate_all.py`
- `experiments/bpgs_analysis/analysis/scripts/generate_figures.py`
- `experiments/bpgs_analysis/analysis/scripts/generate_tables.py`
- `experiments/bpgs_analysis/analysis/scripts/generate_stats.py`
