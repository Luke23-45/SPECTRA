"""
Generate statistical comparison reports for the BPGS paper.

Usage::

    python -m analysis.scripts.generate_stats [--data-root PATH] [--output-dir PATH]

Produces JSON files with pairwise comparisons (bootstrap CIs, Cohen's d,
Wilcoxon tests) for key method pairs in a modular directory structure.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from analysis.common.constants import (
    NYUV2_METHODS,
    NYUV2_TABLE_METRICS,
    SEEDS,
    display_name,
)
from analysis.common.io import resolve_data_root
from analysis.common.metrics import extract_final_epoch
from analysis.common.stats import compare_methods
from analysis.studies.nyuv2.extract import extract_nyuv2_data

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def generate_nyuv2_stats(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate pairwise statistical comparisons for NYUv2."""
    if data_root is None:
        data_root = resolve_data_root()

    data = extract_nyuv2_data(data_root)
    final_epoch = data["final_epoch"]

    metrics = [m[0] for m in NYUV2_TABLE_METRICS]
    bpgs_method = "bpgs"
    baseline_methods = [m for m in NYUV2_METHODS if m != bpgs_method]

    comparisons: List[Dict[str, Any]] = []

    for baseline in baseline_methods:
        for metric in metrics:
            # Gather values per seed
            bpgs_vals = []
            base_vals = []

            for seed in SEEDS:
                if seed in final_epoch.get(bpgs_method, {}):
                    series = final_epoch[bpgs_method][seed]
                    if metric in series.index:
                        bpgs_vals.append(float(series[metric]))

                if seed in final_epoch.get(baseline, {}):
                    series = final_epoch[baseline][seed]
                    if metric in series.index:
                        base_vals.append(float(series[metric]))

            if len(bpgs_vals) >= 2 and len(base_vals) >= 2:
                result = compare_methods(
                    bpgs_vals,
                    base_vals,
                    method_a=bpgs_method,
                    method_b=baseline,
                    metric=metric,
                )
                comparisons.append(result.to_dict())

    report = {
        "study": "nyuv2",
        "primary_method": bpgs_method,
        "baseline_methods": baseline_methods,
        "metrics": metrics,
        "n_seeds": len(SEEDS),
        "comparisons": comparisons,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "nyuv2_pairwise_stats.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, cls=_NumpyEncoder)
        logger.info("Saved NYUv2 stats: %s", out_path)

    return report


def main(args: argparse.Namespace) -> None:
    """Generate all statistical reports into modular subdirectories."""
    data_root = Path(args.data_root) if args.data_root else resolve_data_root()
    base = Path(args.output_dir)

    logger.info("Data root: %s", data_root)
    logger.info("Output dir: %s", base)

    # NYUv2 Stats
    nyuv2_stats_dir = base / "nyuv2" / "stats"
    nyuv2_stats_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating NYUv2 pairwise statistics → %s", nyuv2_stats_dir)
    report = generate_nyuv2_stats(data_root, nyuv2_stats_dir)

    n_sig = sum(
        1 for c in report["comparisons"]
        if c.get("wilcoxon_significant", False)
    )
    logger.info(
        "NYUv2 stats: %d comparisons, %d significant at α=0.05",
        len(report["comparisons"]),
        n_sig,
    )

    logger.info("All statistical reports generated successfully.")


def cli() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate statistical comparison reports.",
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
        help="Base directory for saving JSON reports.",
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
