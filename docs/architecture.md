# Architecture

This document describes the repository structure in terms of executable responsibilities, not just directory names.

## Package Map

```text
spectra/
|-- architectures/
|-- backbones/
|-- baselines/
|-- core/
|-- data/
|-- empirical/
|-- engine/
|-- evaluation/
|-- heads/
|-- modules/
|-- train/
|-- utils/
`-- visualization/
```

## Primary Entry Points

| Path | Responsibility |
|------|----------------|
| `scripts/run_training.py` | Standard Hydra training entrypoint rooted at `configs/` |
| `scripts/run_empirical_experiment.py` | Empirical synthetic benchmark entrypoint rooted at `configs/empirical/` |
| `experiments/bpgs_study/run.py` | Paper study planner and launcher |
| `experiments/bpgs_analysis/analysis/scripts/*.py` | Post-run analysis and paper asset generation |

## Standard Training Path

The standard training flow is:

1. `scripts/run_training.py` composes the Hydra config.
2. `spectra.train.runner.execute_training_mission(...)` builds the datamodule, engine, module, callbacks, loggers, and Lightning trainer.
3. `spectra.data.datamodule.SPECTRADataModule` dispatches to the selected dataset implementation.
4. The selected module in `spectra/modules/` runs the task heads and logs task-specific metrics.

Key implementation files:

- `scripts/run_training.py`
- `spectra/train/runner.py`
- `spectra/train/preflight.py`
- `spectra/data/datamodule.py`

## Training Model Stack

The training stack is split across the following areas:

- backbones in `spectra/backbones/`
- head builders in `spectra/architectures/` and `spectra/heads/`
- Lightning modules in `spectra/modules/`
- optimization engines in `spectra/engine/optimizers/`
- weighting methods in `spectra/baselines/`

Important architectural distinction:

- most methods run through `StandardEngine`
- `pcgrad` uses `PCGradEngine`
- `bpgs` uses `BPGSEngine`

That selection is driven by the composed method config and by logic in `spectra/train/runner.py`.

## Dataset Dispatch

All standard training datasets are routed through one dispatcher:

- `spectra/data/datamodule.py`

That datamodule selects one of:

- `spectra/data/synthetic.py`
- `spectra/data/nyuv2/dataset.py`
- `spectra/data/yeast/dataset.py`
- `spectra/data/rf1/dataset.py`
- `spectra/data/qm9/dataset.py`
- `spectra/data/clinical/dataset.py`

The docs should treat this datamodule as the canonical source of dataset dispatch behavior.

## Empirical Synthetic Benchmark Stack

The empirical benchmark path is separate from the standard training path.

The flow is:

1. `scripts/run_empirical_experiment.py` composes `configs/empirical/`.
2. `spectra.empirical.utils.config_utils.validate_empirical_config(...)` checks the composed config.
3. `spectra.empirical.utils.runtime.EmpiricalRunContext` creates the run directory and metadata.
4. The selected experiment calls into the method-comparison harness and one of the generator families under `spectra/empirical/generators/`.

Key implementation files:

- `scripts/run_empirical_experiment.py`
- `spectra/empirical/utils/config_utils.py`
- `spectra/empirical/utils/runtime.py`
- `spectra/empirical/experiments/common.py`
- `spectra/empirical/runners/method_comparison.py`

## Study Execution Layer

The paper study layer does not implement algorithms. It schedules existing entrypoints.

Repository layout:

```text
experiments/bpgs_study/
|-- ablation/
|-- stress/
|-- real_data/
|-- optional/
`-- common/
```

Responsibilities:

- group directories own study-specific planning
- `common/` provides shared data structures, path helpers, cataloging, and validation

This is a planner/orchestration layer over:

- `scripts.run_training`
- `scripts.run_empirical_experiment`

## Analysis Layer

The paper analysis layer is separate again:

```text
experiments/bpgs_analysis/analysis/
|-- common/
|-- configs/
|-- scripts/
`-- studies/
```

Responsibilities:

- `common/`: shared extraction, metrics, path resolution, LaTeX, style
- `scripts/`: top-level analysis CLIs
- `studies/`: study-family-specific extraction and rendering

The analysis code reads finished outputs; it is not part of the training runtime.

## Reproducibility Artifacts

For standard training runs, reproducibility artifacts are created in:

- `spectra/train/artifacts.py`

For empirical runs, run metadata and resolved config persistence are handled by:

- `spectra/empirical/utils/runtime.py`

For study-level orchestration paths, output roots are defined by:

- `experiments/bpgs_study/common/paths.py`
