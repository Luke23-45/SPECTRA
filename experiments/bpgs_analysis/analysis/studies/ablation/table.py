"""
Ablation study LaTeX table generation.

Produces the ablation comparison table showing BPGS variants vs
Kendall baseline on NYUv2 (mIoU, Abs Rel, RMSE, Angle, Within 11.25°,
Total Loss).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from analysis.common.constants import (
    ABLATION_METHODS,
    NYUV2_TABLE_METRICS,
    MetricDir,
    SEEDS,
)
from analysis.common.latex import build_results_table, save_latex_table
from analysis.studies.ablation.extract import extract_ablation_data

logger = logging.getLogger(__name__)


def generate_ablation_table(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    precision: int = 3,
    standalone: bool = False,
) -> str:
    """Generate the ablation results table.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    output_dir : Path, optional
        Directory for saving ``.tex`` file.  If ``None``, does not save.
    precision : int
        Decimal places for numeric values.
    standalone : bool
        If ``True``, the ``.tex`` file is self-compilable.

    Returns
    -------
    str
        LaTeX table source code.
    """
    data = extract_ablation_data(data_root)

    # Build column name and direction mappings
    column_names = {m[0]: m[1] for m in NYUV2_TABLE_METRICS}
    column_dirs = {m[0]: m[2] for m in NYUV2_TABLE_METRICS}
    metrics = [m[0] for m in NYUV2_TABLE_METRICS]

    tex = build_results_table(
        agg_data=data["aggregated"],
        metrics=metrics,
        column_names=column_names,
        column_directions=column_dirs,
        caption=(
            "Ablation study: BPGS variants vs.\\ Kendall uncertainty "
            "weighting on NYUv2. All methods trained for 60 epochs. "
            "Mean $\\pm$ std over 3 seeds."
        ),
        label="tab:ablation",
        precision=precision,
        highlight_method="bpgs_batch_aware_fixed",
    )

    if output_dir is not None:
        save_latex_table(
            tex,
            Path(output_dir) / "ablation_table.tex",
            standalone_header=standalone,
        )

    return tex
