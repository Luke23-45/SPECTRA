# Analysis

The paper analysis pipeline lives under `experiments/bpgs_analysis/analysis/`. It is a separate post-processing system that reads completed study outputs and produces figures, tables, and selected statistical summaries.

## Layout

```text
experiments/bpgs_analysis/analysis/
|-- common/
|-- configs/
|-- docs/
|-- scripts/
`-- studies/
    |-- ablation/
    |-- full_data/
    |-- nyuv2/
    `-- stress/
```

### Responsibilities

- `common/`: shared I/O, path resolution, metric direction logic, statistical utilities, LaTeX helpers, and plotting style
- `configs/`: analysis-side OmegaConf configuration
- `scripts/`: argparse entrypoints that manage analysis generation
- `studies/`: study-specific extraction, plotting, and table logic

## How to Run It

The analysis modules import `analysis.*`, so run them from `experiments/bpgs_analysis/`:

```bash
cd experiments/bpgs_analysis
python -m analysis.scripts.generate_all
```

Individual entrypoints:

```bash
python -m analysis.scripts.generate_figures
python -m analysis.scripts.generate_tables
python -m analysis.scripts.generate_stats
python -m analysis.scripts.assemble_paper_assets
```

## What Each Script Actually Produces

### `generate_all.py`

Defined in `experiments/bpgs_analysis/analysis/scripts/generate_all.py`.

Purpose:

- runs tables
- runs figures
- runs statistics

Default output root:

- `analysis/outputs/paper_final`

### `generate_figures.py`

Defined in `experiments/bpgs_analysis/analysis/scripts/generate_figures.py`.

Purpose:

- ablation figures
- NYUv2 figures
- Yeast figures
- RF1 figures
- stress figures for scale, rescaling, combined summaries, and heterogeneous regimes

Default output root:

- `analysis/outputs`

### `generate_tables.py`

Defined in `experiments/bpgs_analysis/analysis/scripts/generate_tables.py`.

Purpose:

- ablation table
- NYUv2 table
- Yeast table
- RF1 table
- stress tables for scale, rescaling, degradation, and heterogeneous regimes

Default output root:

- `analysis/outputs`

### `generate_stats.py`

Defined in `experiments/bpgs_analysis/analysis/scripts/generate_stats.py`.

Purpose:

- currently generates pairwise statistical comparisons only for the NYUv2 benchmark

This is important for honesty in the paper workflow: the repository does not currently provide a general statistics generator for every study family through this script.

Default output root:

- `analysis/outputs`

### `assemble_paper_assets.py`

Defined in `experiments/bpgs_analysis/analysis/scripts/assemble_paper_assets.py`.

Purpose:

- copies selected main figures
- copies selected supplementary figures
- copies selected tables
- writes a bundle manifest

## Study Modules

Each study family has its own extraction and reporting code:

- ablation: `analysis/studies/ablation/`
- NYUv2: `analysis/studies/nyuv2/`
- full-data Yeast/RF1: `analysis/studies/full_data/`
- stress: `analysis/studies/stress/`

Most study packages follow this split:

- `extract.py`: load and normalize run outputs
- `figures.py`: generate plots
- `table.py`: generate LaTeX tables

Additional NYUv2-specific metric handling is implemented in:

- `analysis/studies/nyuv2/delta_m.py`

## Configuration

The central analysis config is:

- `experiments/bpgs_analysis/analysis/configs/base.yaml`

That file defines:

- study output roots
- study method lists
- seed lists
- formatting settings
- export settings
- stress-study metadata

The current file also contains legacy naming and path comments, so any paper claim about analysis paths should be grounded in the executable scripts and `analysis/common/io.py`, not in comments alone.

## Data Discovery

Path resolution and run discovery are implemented in:

- `analysis/common/io.py`
- `analysis/common/config.py`
- `analysis/common/constants.py`

These modules define how the analysis pipeline finds:

- ablation outputs
- NYUv2 outputs
- Yeast outputs
- RF1 outputs
- stress outputs

If the result directory layout changes, these files must be updated before the docs are updated.

## Paper Asset Manifest

The analysis-side asset manifest tracked in the repository is:

- `experiments/bpgs_analysis/analysis/docs/final_paper_asset_manifest.md`

The assembled export bundle created by `assemble_paper_assets.py` also writes a generated `README.md` inside the assembled output directory.
