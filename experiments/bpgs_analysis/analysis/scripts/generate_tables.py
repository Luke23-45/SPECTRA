"""
Generate all LaTeX tables for the BPGS paper.

Usage::

    python -m analysis.scripts.generate_tables [--data-root PATH] [--output-dir PATH]

Produces .tex files in a modular directory structure organised by
experiment type and sub-experiment.

Output structure::

    {output-dir}/
    ├── ablation/tables/
    ├── nyuv2/tables/
    ├── full_data/
    │   ├── yeast/tables/
    │   └── river_flow/tables/
    └── stress/
        ├── scale/tables/
        ├── rescaling/tables/
        ├── heterogeneous/tables/
        └── combined/tables/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis.common.io import resolve_data_root

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate all tables into modular subdirectories."""
    data_root = Path(args.data_root) if args.data_root else resolve_data_root()
    base = Path(args.output_dir)

    standalone = args.standalone
    precision = args.precision

    logger.info("Data root: %s", data_root)
    logger.info("Output dir: %s", base)

    # Ablation
    abl_tables = base / "ablation" / "tables"
    abl_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating ablation table → %s", abl_tables)
    from analysis.studies.ablation.table import generate_ablation_table
    generate_ablation_table(
        data_root, abl_tables, precision=precision, standalone=standalone,
    )

    # NYUv2
    nyuv2_tables = base / "nyuv2" / "tables"
    nyuv2_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating NYUv2 table → %s", nyuv2_tables)
    from analysis.studies.nyuv2.table import generate_nyuv2_table
    generate_nyuv2_table(
        data_root, nyuv2_tables, precision=precision, standalone=standalone,
    )

    # Full-data: Yeast
    yeast_tables = base / "full_data" / "yeast" / "tables"
    yeast_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Yeast table → %s", yeast_tables)
    from analysis.studies.full_data.table import generate_yeast_table
    generate_yeast_table(
        data_root, yeast_tables, precision=precision, standalone=standalone,
    )

    # Full-data: River Flow (RF1)
    rf1_tables = base / "full_data" / "river_flow" / "tables"
    rf1_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating RF1 table → %s", rf1_tables)
    from analysis.studies.full_data.table import generate_rf1_table
    generate_rf1_table(
        data_root, rf1_tables, precision=precision, standalone=standalone,
    )

    # Stress: Scale
    scale_tables = base / "stress" / "scale" / "tables"
    scale_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating scale stress table → %s", scale_tables)
    from analysis.studies.stress.table import generate_scale_stress_table
    generate_scale_stress_table(
        data_root, scale_tables, "scale",
        precision=precision, standalone=standalone,
    )

    # Stress: Rescaling
    rescale_tables = base / "stress" / "rescaling" / "tables"
    rescale_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating rescaling stress table → %s", rescale_tables)
    generate_scale_stress_table(
        data_root, rescale_tables, "rescaling",
        precision=precision, standalone=standalone,
    )

    # Stress: Combined (degradation)
    stress_combined_tables = base / "stress" / "combined" / "tables"
    stress_combined_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating degradation table → %s", stress_combined_tables)
    from analysis.studies.stress.table import generate_degradation_table
    generate_degradation_table(
        data_root, stress_combined_tables, standalone=standalone,
    )

    # Stress: Heterogeneous
    hetero_tables = base / "stress" / "heterogeneous" / "tables"
    hetero_tables.mkdir(parents=True, exist_ok=True)

    logger.info("Generating heterogeneous table → %s", hetero_tables)
    from analysis.studies.stress.table import generate_heterogeneous_table
    generate_heterogeneous_table(
        data_root, hetero_tables, precision=precision, standalone=standalone,
    )

    logger.info("All tables generated successfully.")


def cli() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate all LaTeX tables for the BPGS paper.",
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
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for numeric values.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Produce self-compilable .tex files.",
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
