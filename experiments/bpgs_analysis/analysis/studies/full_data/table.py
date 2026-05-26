"""
Full-data regime LaTeX table generation.

Produces:
- ``yeast_table.tex``: 6 methods × (macro-F1, micro-F1, hamming accuracy).
- ``rf1_table.tex``: 6 methods × (R², RMSE, MAE).
- ``full_data_combined_table.tex``: Combined Yeast + RF1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.common.constants import (
    FULL_DATA_METHODS,
    MetricDir,
    SEEDS,
    display_name,
    sort_methods,
)
from analysis.common.latex import (
    build_results_table,
    format_mean_std,
    save_latex_table,
    to_booktabs,
)
from analysis.studies.full_data.extract import (
    RF1_TABLE_METRICS,
    YEAST_TABLE_METRICS,
    extract_full_data,
)

logger = logging.getLogger(__name__)


def generate_yeast_table(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    precision: int = 3,
    standalone: bool = False,
) -> str:
    """Generate the Yeast results table."""
    data = extract_full_data(data_root, "yeast")

    column_names = {m[0]: m[1] for m in YEAST_TABLE_METRICS}
    column_dirs = {m[0]: m[2] for m in YEAST_TABLE_METRICS}
    metrics = [m[0] for m in YEAST_TABLE_METRICS]

    tex = build_results_table(
        agg_data=data["aggregated"],
        metrics=metrics,
        column_names=column_names,
        column_directions=column_dirs,
        caption=(
            "Yeast multi-label classification (14 tasks). "
            "Final-epoch task metrics reported as mean $\\pm$ std over 3 seeds."
        ),
        label="tab:yeast",
        precision=precision,
        highlight_method="bpgs",
    )

    if output_dir is not None:
        save_latex_table(
            tex,
            Path(output_dir) / "yeast_table.tex",
            standalone_header=standalone,
        )

    return tex


def generate_rf1_table(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    precision: int = 3,
    standalone: bool = False,
) -> str:
    """Generate the RF1 results table."""
    data = extract_full_data(data_root, "rf1")

    column_names = {m[0]: m[1] for m in RF1_TABLE_METRICS}
    column_dirs = {m[0]: m[2] for m in RF1_TABLE_METRICS}
    metrics = [m[0] for m in RF1_TABLE_METRICS]

    tex = build_results_table(
        agg_data=data["aggregated"],
        metrics=metrics,
        column_names=column_names,
        column_directions=column_dirs,
        caption=(
            "RF1 multi-target regression (8 flow sites). "
            "Final-epoch task metrics reported as mean $\\pm$ std over 3 seeds."
        ),
        label="tab:rf1",
        precision=precision,
        highlight_method="bpgs",
    )

    if output_dir is not None:
        save_latex_table(
            tex,
            Path(output_dir) / "rf1_table.tex",
            standalone_header=standalone,
        )

    return tex

