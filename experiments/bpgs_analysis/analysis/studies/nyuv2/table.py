"""
NYUv2 benchmark LaTeX table generation.

Produces the main results table: BPGS vs baselines (4 methods × 6 metrics + ΔM).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from analysis.common.constants import NYUV2_TABLE_METRICS, MetricDir
from analysis.common.latex import build_results_table, save_latex_table
from analysis.studies.nyuv2.extract import extract_nyuv2_data

logger = logging.getLogger(__name__)


def generate_nyuv2_table(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    precision: int = 3,
    standalone: bool = False,
    baseline_method: str = "static",
) -> str:
    """Generate the main NYUv2 results table.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    output_dir : Path, optional
        Directory for saving ``.tex`` file.
    precision : int
        Decimal places.
    standalone : bool
        Self-compilable LaTeX output.
    baseline_method : str
        Baseline for ΔM computation.

    Returns
    -------
    str
        LaTeX table source code.
    """
    data = extract_nyuv2_data(data_root, baseline_method=baseline_method)

    column_names = {m[0]: m[1] for m in NYUV2_TABLE_METRICS}
    column_dirs = {m[0]: m[2] for m in NYUV2_TABLE_METRICS}
    metrics = [m[0] for m in NYUV2_TABLE_METRICS]

    tex = build_results_table(
        agg_data=data["aggregated"],
        metrics=metrics,
        column_names=column_names,
        column_directions=column_dirs,
        caption=(
            "NYUv2 dense prediction benchmark. All methods trained for "
            "120 epochs. Final-epoch metrics reported as mean $\\pm$ std "
            "over 3 seeds. $\\Delta_M$ computed against Static baseline."
        ),
        label="tab:nyuv2_main",
        precision=precision,
        highlight_method="bpgs",
        delta_m=data["delta_m"],
    )

    if output_dir is not None:
        save_latex_table(
            tex,
            Path(output_dir) / "nyuv2_main_table.tex",
            standalone_header=standalone,
        )

    return tex
