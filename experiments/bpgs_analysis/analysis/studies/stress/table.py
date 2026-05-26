"""
Stress test LaTeX table generation.

Produces:
- ``scale_stress_table.tex``: Methods x scales (x1..x1000) Macro Score.
- ``degradation_table.tex``: Relative change from x1 to x1000.
- ``heterogeneous_table.tex``: Methods x regimes with Macro / Worst scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.common.constants import (
    STRESS_HETERO_METHODS,
    STRESS_REGIMES,
    STRESS_SCALE_METHODS,
    STRESS_SCALES,
    display_name,
    sort_methods,
)
from analysis.common.latex import _apply_bold, format_mean_std, save_latex_table
from analysis.studies.stress.extract import (
    extract_heterogeneous_stress,
    extract_scale_stress,
)

logger = logging.getLogger(__name__)


def generate_scale_stress_table(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment: str = "scale",
    *,
    precision: int = 3,
    standalone: bool = False,
) -> str:
    """Generate the scale stress results table."""
    data = extract_scale_stress(data_root, experiment)
    aggregated = data["aggregated"]

    methods = sort_methods(list(STRESS_SCALE_METHODS))
    scales = list(STRESS_SCALES)
    scale_headers = [f"$\\times{{{s}}}$" for s in scales]

    exp_label = "02" if experiment == "scale" else "06"
    exp_name = (
        "Synthetic scale stress" if experiment == "scale"
        else "Pure loss rescaling"
    )

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{{exp_name} ({exp_label}): Macro Score across "
        f"scale factors. Mean $\\pm$ std over 3 seeds.}}"
    )
    lines.append(f"\\label{{tab:stress_{experiment}}}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{l " + " ".join(["c"] * len(scales)) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(scale_headers) + " \\\\")
    lines.append("\\midrule")

    all_means: Dict[int, List[float]] = {s: [] for s in scales}
    cell_data: List[List[str]] = []

    for method in methods:
        cells: List[str] = []
        for scale in scales:
            method_data = aggregated.get(scale, {}).get(method, {})
            if "macro_score" in method_data:
                mean, std = method_data["macro_score"]
                cells.append(format_mean_std(mean, std, precision=precision))
                all_means[scale].append(mean)
            else:
                cells.append("—")
                all_means[scale].append(np.nan)
        cell_data.append(cells)

    for j, scale in enumerate(scales):
        vals = np.array(all_means[scale], dtype=np.float64)
        if np.isfinite(vals).sum() >= 2:
            best_idx = int(np.nanargmax(vals))
            cell_data[best_idx][j] = _apply_bold(cell_data[best_idx][j])

    for i, method in enumerate(methods):
        lines.append(f"{display_name(method)} & " + " & ".join(cell_data[i]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    if output_dir is not None:
        save_latex_table(
            tex,
            Path(output_dir) / f"scale_stress_{experiment}_table.tex",
            standalone_header=standalone,
        )
    return tex


def generate_degradation_table(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    precision: int = 1,
    standalone: bool = False,
) -> str:
    """Generate a degradation comparison table."""
    scale_data = extract_scale_stress(data_root, "scale")
    rescale_data = extract_scale_stress(data_root, "rescaling")

    methods = sort_methods(list(STRESS_SCALE_METHODS))
    scale_degs = scale_data["degradation"]
    rescale_degs = rescale_data["degradation"]

    scale_vals = np.array(
        [scale_degs.get(method, np.nan) for method in methods],
        dtype=np.float64,
    )
    rescale_vals = np.array(
        [rescale_degs.get(method, np.nan) for method in methods],
        dtype=np.float64,
    )
    best_scale_idx = int(np.nanargmin(scale_vals)) if np.isfinite(scale_vals).any() else None
    best_rescale_idx = int(np.nanargmin(rescale_vals)) if np.isfinite(rescale_vals).any() else None

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Relative Macro Score change (\\%) from "
        "$\\times 1 \\to \\times 1000$. Positive values indicate degradation; "
        "negative values indicate improvement.}"
    )
    lines.append("\\label{tab:degradation}")
    lines.append("\\begin{tabular}{l c c}")
    lines.append("\\toprule")
    lines.append("Method & Scale Stress & Loss Rescaling \\\\")
    lines.append("\\midrule")

    for idx, method in enumerate(methods):
        s_deg = scale_degs.get(method, np.nan)
        r_deg = rescale_degs.get(method, np.nan)
        s_pct = 0.0 if not np.isnan(s_deg) and abs(s_deg) < 5e-4 else s_deg * 100
        r_pct = 0.0 if not np.isnan(r_deg) and abs(r_deg) < 5e-4 else r_deg * 100
        s_str = f"${s_pct:.{precision}f}\\%$" if not np.isnan(s_deg) else "—"
        r_str = f"${r_pct:.{precision}f}\\%$" if not np.isnan(r_deg) else "—"
        if best_scale_idx is not None and idx == best_scale_idx:
            s_str = _apply_bold(s_str)
        if best_rescale_idx is not None and idx == best_rescale_idx:
            r_str = _apply_bold(r_str)
        lines.append(f"{display_name(method)} & {s_str} & {r_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    if output_dir is not None:
        save_latex_table(
            tex,
            Path(output_dir) / "degradation_table.tex",
            standalone_header=standalone,
        )
    return tex


def generate_heterogeneous_table(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    precision: int = 3,
    standalone: bool = False,
) -> str:
    """Generate the heterogeneous stress results table."""
    data = extract_heterogeneous_stress(data_root)
    aggregated = data["aggregated"]

    methods = sort_methods(list(STRESS_HETERO_METHODS))
    regimes = list(STRESS_REGIMES)
    metrics = [
        ("macro_score", "Macro"),
        ("worst_task_score", "Worst"),
    ]

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Heterogeneous mixed stress (07): Macro Score and "
        "Worst-Task Score across regimes. Mean $\\pm$ std over 3 seeds.}"
    )
    lines.append("\\label{tab:stress_heterogeneous}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append(
        "\\begin{tabular}{l " + " ".join(["c"] * (len(regimes) * len(metrics))) + "}"
    )
    lines.append("\\toprule")
    lines.append(
        "Method & "
        + " & ".join(
            [f"\\multicolumn{{2}}{{c}}{{{regime.capitalize()}}}" for regime in regimes]
        )
        + " \\\\"
    )
    lines.append("& " + " & ".join([label for _ in regimes for _, label in metrics]) + " \\\\")
    lines.append("\\midrule")

    ordered_cols = [(regime, metric) for regime in regimes for metric, _ in metrics]
    all_means: Dict[Tuple[str, str], List[float]] = {key: [] for key in ordered_cols}
    cell_data: List[List[str]] = []

    for method in methods:
        cells: List[str] = []
        for regime, metric in ordered_cols:
            method_data = aggregated.get(regime, {}).get(method, {})
            if metric in method_data:
                mean, std = method_data[metric]
                cells.append(format_mean_std(mean, std, precision=precision))
                all_means[(regime, metric)].append(mean)
            else:
                cells.append("—")
                all_means[(regime, metric)].append(np.nan)
        cell_data.append(cells)

    for j, key in enumerate(ordered_cols):
        vals = np.array(all_means[key], dtype=np.float64)
        if np.isfinite(vals).sum() >= 2:
            best_idx = int(np.nanargmax(vals))
            cell_data[best_idx][j] = _apply_bold(cell_data[best_idx][j])

    for i, method in enumerate(methods):
        lines.append(f"{display_name(method)} & " + " & ".join(cell_data[i]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    if output_dir is not None:
        save_latex_table(
            tex,
            Path(output_dir) / "heterogeneous_table.tex",
            standalone_header=standalone,
        )
    return tex
