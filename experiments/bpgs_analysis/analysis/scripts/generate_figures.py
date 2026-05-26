"""
Generate all publication figures for the BPGS paper.

Usage::

    python -m analysis.scripts.generate_figures [--data-root PATH] [--output-dir PATH]

Produces PDF and PNG versions of all figures in a modular directory
structure organised by experiment type and sub-experiment.

Output structure::

    {output-dir}/
    ├── ablation/figures/{pdf,png}/
    ├── nyuv2/figures/{pdf,png}/
    ├── full_data/
    │   ├── yeast/figures/{pdf,png}/
    │   ├── river_flow/figures/{pdf,png}/
    │   └── combined/figures/{pdf,png}/
    └── stress/
        ├── heterogeneous/figures/{pdf,png}/
        └── combined/figures/{pdf,png}/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis.common.io import resolve_data_root

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate all figures into modular subdirectories."""
    data_root = Path(args.data_root) if args.data_root else resolve_data_root()
    base = Path(args.output_dir)

    logger.info("Data root: %s", data_root)
    logger.info("Output dir: %s", base)

    # Ablation
    ablation_fig = base / "ablation" / "figures"
    ablation_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.ablation.figures import (
        generate_training_curves as abl_curves,
        generate_weight_dynamics as abl_weights,
        generate_per_task_val_curves as abl_per_task,
    )
    logger.info("Generating ablation figures → %s", ablation_fig)
    abl_curves(data_root, ablation_fig)
    abl_weights(data_root, ablation_fig)
    abl_per_task(data_root, ablation_fig)

    # NYUv2
    nyuv2_fig = base / "nyuv2" / "figures"
    nyuv2_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.nyuv2.figures import (
        generate_task_training_curves as nyuv2_curves,
        generate_performance_comparison as nyuv2_perf,
        generate_relative_improvement as nyuv2_rel,
        generate_delta_m_comparison as nyuv2_dm,
        generate_performance_heatmap as nyuv2_heat,
    )
    logger.info("Generating NYUv2 figures → %s", nyuv2_fig)
    nyuv2_curves(data_root, nyuv2_fig)
    nyuv2_perf(data_root, nyuv2_fig)
    nyuv2_rel(data_root, nyuv2_fig)
    nyuv2_dm(data_root, nyuv2_fig)
    nyuv2_heat(data_root, nyuv2_fig)

    # Full-data: Yeast
    yeast_fig = base / "full_data" / "yeast" / "figures"
    yeast_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.full_data.figures import (
        generate_yeast_training_curves,
    )
    logger.info("Generating Yeast training curves → %s", yeast_fig)
    generate_yeast_training_curves(data_root, yeast_fig)

    # Full-data: River Flow (RF1)
    rf1_fig = base / "full_data" / "river_flow" / "figures"
    rf1_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.full_data.figures import (
        generate_rf1_training_curves,
    )
    logger.info("Generating River Flow training curves → %s", rf1_fig)
    generate_rf1_training_curves(data_root, rf1_fig)

    # Stress: Combined
    stress_combined_fig = base / "stress" / "combined" / "figures"
    stress_combined_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.stress.figures import (
        generate_stress_robustness_summary,
        generate_degradation_comparison,
    )
    logger.info("Generating stress combined figures → %s", stress_combined_fig)
    generate_stress_robustness_summary(data_root, stress_combined_fig)
    generate_degradation_comparison(data_root, stress_combined_fig)

    # Stress: Scale
    scale_fig = base / "stress" / "scale" / "figures"
    scale_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.stress.figures import (
        generate_scale_stress_curves,
    )
    logger.info("Generating scale stress figures → %s", scale_fig)
    generate_scale_stress_curves(data_root, scale_fig)

    # Stress: Rescaling
    rescaling_fig = base / "stress" / "rescaling" / "figures"
    rescaling_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.stress.figures import (
        generate_rescaling_stress_curves,
    )
    logger.info("Generating rescaling figures → %s", rescaling_fig)
    generate_rescaling_stress_curves(data_root, rescaling_fig)

    # Stress: Heterogeneous
    hetero_fig = base / "stress" / "heterogeneous" / "figures"
    hetero_fig.mkdir(parents=True, exist_ok=True)

    from analysis.studies.stress.figures import (
        generate_heterogeneous_panel,
    )
    logger.info("Generating heterogeneous figures → %s", hetero_fig)
    generate_heterogeneous_panel(data_root, hetero_fig)

    logger.info("All figures generated successfully.")


def cli() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate all publication figures for the BPGS paper.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to bpgs_experiment/data/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/outputs",
        help="Base directory for all outputs.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args)


if __name__ == "__main__":
    cli()
