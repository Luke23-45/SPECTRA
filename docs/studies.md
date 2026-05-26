# Studies

The paper-oriented study runner is `experiments/bpgs_study/run.py`. It is a planner and launcher for named study definitions; it is not a separate training framework.

The study definitions are loaded through:

- `experiments/bpgs_study/common/catalog.py`
- `experiments/bpgs_study/common/config_loader.py`
- `experiments/bpgs_study/common/specs.py`

## What the Study Runner Does

Given a study name, the runner:

1. resolves the registered `StudySpec`
2. validates the request
3. builds a stage list
4. either prints those stages with `--dry-run` or executes them in order

The stage builders are group-specific:

- ablation: `experiments/bpgs_study/ablation/`
- stress: `experiments/bpgs_study/stress/`
- real_data: `experiments/bpgs_study/real_data/`
- optional: `experiments/bpgs_study/optional/`

## Common Commands

```bash
python experiments/bpgs_study/run.py --list
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --dry-run
python experiments/bpgs_study/run.py --study 03_yeast_regime_check --variant bpgs --seed 42
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --start-from 2
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --keep-going
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --skip-prep
```

## Override Syntax

Study overrides are parsed by `OverrideSpec` in `experiments/bpgs_study/common/specs.py`.

Accepted forms:

- `key=value`
- `method_name:key=value`

Examples:

```bash
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --set bpgs:method.s_mode=stateless
python experiments/bpgs_study/run.py --study 03_yeast_regime_check --set train.batch_size=8
```

## Registered Study Groups

### Ablation

| Study | Type | Definition file | Purpose |
|------|------|------------------|---------|
| `01_nyuv2_ablation` | Training | `experiments/bpgs_study/ablation/configs/01_nyuv2_ablation.yaml` | Subset-based NYUv2 ablation of canonical BPGS, nearby BPGS variants, and Kendall |

This study may include a preparation stage that materializes deterministic NYUv2 subset files through `experiments/bpgs_study/nyuv2_subsets.py`.

### Stress

| Study | Type | Definition file | Purpose |
|------|------|------------------|---------|
| `02_synthetic_scale_stress` | Empirical | `experiments/bpgs_study/stress/configs/02_synthetic_scale_stress.yaml` | Controlled imbalance-ratio sweep built on `exp_03_imbalance_robustness` |
| `06_pure_loss_rescaling` | Empirical | `experiments/bpgs_study/stress/configs/06_pure_loss_rescaling.yaml` | Loss-rescaling stress built on `exp_07_pure_loss_rescaling` |
| `07_heterogeneous_mixed_stress` | Empirical | `experiments/bpgs_study/stress/configs/07_heterogeneous_mixed_stress.yaml` | Heterogeneous mixed-task stress built on `exp_08_heterogeneous_regime` |

These studies dispatch to `scripts/run_empirical_experiment.py`, not to `scripts/run_training.py`.

### Real Data

| Study | Type | Definition file | Purpose |
|------|------|------------------|---------|
| `03_yeast_regime_check` | Training | `experiments/bpgs_study/real_data/configs/03_yeast_regime_check.yaml` | Multi-seed Yeast support study |
| `08_rf1_regime_check` | Training | `experiments/bpgs_study/real_data/configs/08_rf1_regime_check.yaml` | Multi-seed RF1 support study |

Both files explicitly describe these studies as supporting evidence rather than as replacements for the core synthetic stress story.

### Optional

| Study | Type | Definition file | Purpose |
|------|------|------------------|---------|
| `04_qm9_regime_check` | Training | `experiments/bpgs_study/optional/configs/04_qm9_regime_check.yaml` | Optional QM9 support study |
| `05_nyuv2_full_final` | Training | `experiments/bpgs_study/optional/configs/05_nyuv2_full_final.yaml` | Optional full NYUv2 seeded comparison for the paper table |

These are intentionally marked optional in the study definitions and should be interpreted that way in the paper workflow.

## How Commands Are Built

Training studies construct commands of the form:

```text
python -m scripts.run_training dataset=<dataset> method=<method> ...
```

The builders live in:

- `experiments/bpgs_study/ablation/commands.py`
- `experiments/bpgs_study/real_data/commands.py`
- `experiments/bpgs_study/optional/commands.py`

Stress studies construct commands of the form:

```text
python -m scripts.run_empirical_experiment experiment=<experiment> family=<family> methods.names=[...] ...
```

The builder lives in:

- `experiments/bpgs_study/stress/commands.py`

## Output Layout

The preferred study output root is defined in:

- `experiments/bpgs_study/common/paths.py`

The current preferred root is:

```text
outputs/experiments/bpgs_study/
```

Training study runs are organized as:

```text
outputs/experiments/bpgs_study/<study_name>/<variant_label>/seed_<N>/
```

Empirical stress runs are organized as:

```text
outputs/experiments/bpgs_study/<study_name>/<variant_label>/<method>/seed_<N>/
```

The same paths module also retains a legacy fallback root for older result trees. The repository should not document that legacy path as the primary location.

## Recommended Validation Step

Before running a paper-facing study, inspect the generated plan:

```bash
python experiments/bpgs_study/run.py --study 01_nyuv2_ablation --dry-run
python experiments/bpgs_study/run.py --study 02_synthetic_scale_stress --dry-run
```

This is the most direct way to confirm:

- which entrypoint will be called
- which dataset or empirical family will be used
- which methods and seeds will be executed
- where outputs will be written
