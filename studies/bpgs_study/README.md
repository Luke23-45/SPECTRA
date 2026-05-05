# BPGS Study

This package owns experiment execution for the BPGS study suite.

It is intentionally limited to study-time concerns:

- study definitions and grouping
- **study-owned execution planning** — each study group owns its run logic
- deterministic NYUv2 subset preparation
- routing run outputs into the study output tree

It does not own post-run aggregation, paper tables, cross-study reports, or publication artifacts. Those now live under `analysis/bpgs_study/`.

## Architecture: Study-Owned Execution

**Core principle:** Each study group (ablation, stress, real_data, optional) owns its execution logic completely. The `common/` module provides only data types, paths, config loading, and selection helpers — no behavioral contracts.

- Each study has a `runner.py` class that implements the `StudyRunner` protocol
- Each study has a `commands.py` module with its own command-building logic
- No study is forced to inherit behavior from `common/` — they can customize everything

This enables **method-scoped overrides**: use `method:key=value` syntax to apply overrides only to variants with a specific method. For example:

```bash
python studies/bpgs_study/run.py --study 01_nyuv2_ablation \
    --set bpgs:method.s_mode=stateless \
    --set bpgs:method.init_mode=auto_calibrate
```

The `bpgs:method.s_mode=stateless` override applies only to variants with `method=bpgs` — it will NOT be applied to `kendall` variants (which don't support that parameter).

## Layout

- `common/`
  - `paths.py` - project, study, subset, and study output roots
  - `specs.py` - `StudySpec`, `TrainingVariant`, `EmpiricalVariant`, `OverrideSpec`
  - `config_loader.py` - YAML to `StudySpec`
  - `runtime.py` - `RunRequest`, `Stage`, and selection helpers
  - `validation.py` - generic validation utilities
  - `catalog.py` - study registry using the `StudyRunner` protocol
- `ablation/`
  - `runner.py` - `AblationRunner` class (owns all ablation execution logic)
  - `commands.py` - ablation-specific command builders
  - `configs/` - ablation study definitions
- `stress/`
  - `runner.py` - `StressRunner` class
  - `commands.py` - stress-specific command builders
  - `configs/` - stress study definitions
- `real_data/`
  - `runner.py` - `RealDataRunner` class
  - `commands.py` - real-data-specific command builders
  - `configs/` - real_data study definitions
- `optional/`
  - `runner.py` - `OptionalRunner` class
  - `commands.py` - optional-specific command builders
  - `configs/` - optional study definitions
- `nyuv2_subsets.py`
  - deterministic subset generation for NYUv2 ablations
- `run.py`
  - CLI for executing one registered study
- `definitions.py`
  - compatibility exports for study discovery and study output roots

## Analysis

Post-run analysis now lives here:

- `analysis/bpgs_study/aggregate.py`
- `analysis/bpgs_study/aggregation.py`
- `analysis/bpgs_study/aggregation_v2.py`
- `analysis/bpgs_study/cross_study.py`
- `analysis/bpgs_study/publication.py`
- `analysis/bpgs_study/reproducibility.py`
- `analysis/bpgs_study/statistics.py`

## Commands

List studies:

```bash
python studies/bpgs_study/run.py --list
```

Dry-run one study:

```bash
python studies/bpgs_study/run.py --study 01_nyuv2_ablation --dry-run
```

Dry-run a narrowed ablation slice with **method-scoped overrides**:

```bash
python studies/bpgs_study/run.py --study 01_nyuv2_ablation \
    --variant bpgs_canonical --seed 42 \
    --set bpgs:method.s_mode=stateless \
    --set bpgs:method.init_mode=auto_calibrate \
    --dry-run
```

The `bpgs:` prefix scopes those overrides to only apply to variants with `method=bpgs`.

Dry-run with **global overrides** (applies to all variants):

```bash
python studies/bpgs_study/run.py --study 03_yeast_regime_check \
    --variant bpgs --seed 42 \
    --set train.batch_size=8 \
    --dry-run
```

Aggregate one study:

```bash
python analysis/bpgs_study/aggregate.py --study 06_pure_loss_rescaling
```

Cross-study aggregation:

```bash
python analysis/bpgs_study/aggregate.py --cross-study
```

## Output Roots

Study execution outputs:

```text
outputs/studies/bpgs_study/
```

Analysis outputs:

```text
outputs/analysis/bpgs_study/
```
