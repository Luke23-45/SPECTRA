"""
LaTeX table generation for the BPGS analysis framework.

Produces publication-ready LaTeX tables using ``booktabs`` style,
with support for:
- ``mean ± std`` formatting with configurable precision.
- Bold-best and underline-second highlighting per column.
- Automatic ``\\resizebox`` wrapping for NeurIPS column widths.
- Direct ``.tex`` file export.

All generated tables are standalone-compilable via:

.. code-block:: latex

    \\documentclass{article}
    \\usepackage{booktabs}
    \\begin{document}
    \\input{table.tex}
    \\end{document}

Design notes
------------
* We never use ``tabularx`` or ``longtable``; NeurIPS papers use
  standard ``tabular`` wrapped in ``\\resizebox`` when needed.
* The ``\\pm`` symbol renders as ± in LaTeX math mode.
* Method names in the first column are escaped for LaTeX safety.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from analysis.common.constants import (
    MetricDir,
    display_name,
    get_metric_direction,
    sort_methods,
)

logger = logging.getLogger(__name__)


def format_mean_std(
    mean: float,
    std: float,
    *,
    precision: int = 3,
    use_math: bool = True,
) -> str:
    """Format a mean ± std value for LaTeX.

    Parameters
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation (or SEM).
    precision : int
        Number of decimal places.
    use_math : bool
        If ``True``, wrap in ``$...$`` for LaTeX math mode.

    Returns
    -------
    str
        Formatted string, e.g. ``"$0.287 \\pm 0.003$"``.

    Examples
    --------
    >>> format_mean_std(0.28733, 0.00312, precision=3)
    '$0.287 \\\\pm 0.003$'
    >>> format_mean_std(35.18, 0.42, precision=2)
    '$35.18 \\\\pm 0.42$'
    """
    if np.isnan(mean):
        return "—" if not use_math else "$-$"

    fmt = f"{{:.{precision}f}}"
    mean_s = fmt.format(mean)
    std_s = fmt.format(std)

    if use_math:
        return f"${mean_s} \\pm {std_s}$"
    return f"{mean_s} ± {std_s}"


def format_value(
    value: float,
    *,
    precision: int = 3,
    use_math: bool = True,
) -> str:
    """Format a single numeric value for LaTeX.

    Parameters
    ----------
    value : float
        Value to format.
    precision : int
        Number of decimal places.
    use_math : bool
        If ``True``, wrap in ``$...$``.

    Returns
    -------
    str
    """
    if np.isnan(value):
        return "—" if not use_math else "$-$"
    fmt = f"{{:.{precision}f}}"
    s = fmt.format(value)
    return f"${s}$" if use_math else s


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text.

    Does NOT escape text already containing LaTeX commands
    (detected by presence of backslash).
    """
    if "\\" in text or "$" in text:
        return text  # assume already LaTeX-safe
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def _apply_bold(cell: str) -> str:
    """Wrap a LaTeX cell value in \\textbf{} or \\mathbf{}.

    Handles both math-mode ($...$) and plain-text cells.
    """
    if cell.startswith("$") and cell.endswith("$"):
        inner = cell[1:-1]
        return f"$\\mathbf{{{inner}}}$"
    return f"\\textbf{{{cell}}}"


def _apply_underline(cell: str) -> str:
    """Wrap a LaTeX cell value in \\underline{}."""
    if cell.startswith("$") and cell.endswith("$"):
        inner = cell[1:-1]
        return f"$\\underline{{{inner}}}$"
    return f"\\underline{{{cell}}}"


def to_booktabs(
    data: pd.DataFrame,
    caption: str,
    label: str,
    *,
    column_names: Optional[Dict[str, str]] = None,
    column_directions: Optional[Dict[str, MetricDir]] = None,
    precision: int = 3,
    bold_best: bool = True,
    underline_second: bool = True,
    highlight_method: Optional[str] = None,
    method_col: str = "method",
    resize: bool = True,
    position: str = "t",
    mean_std_columns: Optional[Sequence[str]] = None,
    mean_suffix: str = "_mean",
    std_suffix: str = "_std",
) -> str:
    """Generate a complete LaTeX booktabs table.

    Parameters
    ----------
    data : pd.DataFrame
        Table data.  Should have a ``method`` column (or index) and
        metric columns.  For mean±std display, provide paired columns
        with ``_mean`` and ``_std`` suffixes (or set *mean_std_columns*).
    caption : str
        Table caption text.
    label : str
        LaTeX label (e.g. ``"tab:nyuv2_main"``).
    column_names : dict[str, str], optional
        ``raw_col → display_name`` mapping for column headers.
        If ``None``, columns are displayed as-is.
    column_directions : dict[str, MetricDir], optional
        ``raw_col → direction`` for highlighting.  If ``None``,
        looked up from ``constants.METRIC_DIRECTION``.
    precision : int
        Decimal precision for numeric values.
    bold_best : bool
        Whether to bold the best value per column.
    underline_second : bool
        Whether to underline the second-best value per column.
    highlight_method : str, optional
        If set, highlight this method's row name with ``\\textbf``.
    method_col : str
        Column (or index) containing method identifiers.
    resize : bool
        Whether to wrap in ``\\resizebox{\\columnwidth}{!}{...}``.
    position : str
        Table float position (e.g. ``"t"``, ``"h"``, ``"!ht"``).
    mean_std_columns : sequence of str, optional
        Metrics that have paired ``<col>_mean`` / ``<col>_std``
        columns in the data.  These will be formatted as
        ``mean ± std``.
    mean_suffix, std_suffix : str
        Suffixes identifying mean and std columns.

    Returns
    -------
    str
        Complete LaTeX table source code.
    """
    df = data.copy()

    # If method is the index, move it to a column
    if method_col not in df.columns and df.index.name == method_col:
        df = df.reset_index()

    # Determine display columns
    if mean_std_columns is not None:
        display_cols = list(mean_std_columns)
    else:
        # Auto-detect: find columns ending with mean_suffix
        display_cols = [
            c.replace(mean_suffix, "")
            for c in df.columns
            if c.endswith(mean_suffix)
        ]
        if not display_cols:
            # No mean/std pairs — use all numeric columns except method
            display_cols = [
                c for c in df.columns
                if c != method_col and pd.api.types.is_numeric_dtype(df[c])
            ]

    has_mean_std = any(
        f"{c}{mean_suffix}" in df.columns for c in display_cols
    )

    # Sort methods by canonical order
    if method_col in df.columns:
        ordered = sort_methods(df[method_col].tolist())
        df = df.set_index(method_col).loc[
            [m for m in ordered if m in df[method_col].values]
        ].reset_index()

    # Build header
    col_headers = []
    for col in display_cols:
        if column_names and col in column_names:
            header = column_names[col]
        else:
            header = col.replace("_", " ").replace("/", " ")
        col_headers.append(header)

    # Format cells + track raw values for highlighting
    n_methods = len(df)
    formatted_cells: List[List[str]] = []
    raw_values: Dict[str, List[float]] = {c: [] for c in display_cols}

    for _, row in df.iterrows():
        cells: List[str] = []
        for col in display_cols:
            if has_mean_std and f"{col}{mean_suffix}" in df.columns:
                mean_val = row.get(f"{col}{mean_suffix}", np.nan)
                std_val = row.get(f"{col}{std_suffix}", np.nan)
                if pd.isna(mean_val):
                    cells.append("—")
                    raw_values[col].append(np.nan)
                else:
                    cells.append(
                        format_mean_std(mean_val, std_val, precision=precision)
                    )
                    raw_values[col].append(float(mean_val))
            elif col in df.columns:
                val = row[col]
                if pd.isna(val):
                    cells.append("—")
                    raw_values[col].append(np.nan)
                elif isinstance(val, float):
                    cells.append(format_value(val, precision=precision))
                    raw_values[col].append(float(val))
                else:
                    cells.append(str(val))
                    try:
                        raw_values[col].append(float(val))
                    except (ValueError, TypeError):
                        raw_values[col].append(np.nan)
            else:
                cells.append("—")
                raw_values[col].append(np.nan)

        formatted_cells.append(cells)

    # Apply bold-best / underline-second
    if bold_best and n_methods >= 2:
        for j, col in enumerate(display_cols):
            vals = np.array(raw_values[col], dtype=np.float64)
            valid_mask = ~np.isnan(vals)

            if valid_mask.sum() < 2:
                continue

            # Determine direction
            if column_directions and col in column_directions:
                direction = column_directions[col]
            else:
                try:
                    direction = get_metric_direction(col)
                except KeyError:
                    continue

            if direction == MetricDir.MAX:
                sorted_indices = np.argsort(-vals)
            else:
                sorted_indices = np.argsort(vals)

            # Filter to valid indices only
            sorted_valid = [i for i in sorted_indices if valid_mask[i]]

            if len(sorted_valid) >= 1:
                best_idx = sorted_valid[0]
                formatted_cells[best_idx][j] = _apply_bold(
                    formatted_cells[best_idx][j]
                )

            if underline_second and len(sorted_valid) >= 2:
                second_idx = sorted_valid[1]
                formatted_cells[second_idx][j] = _apply_underline(
                    formatted_cells[second_idx][j]
                )

    # Assemble LaTeX
    n_cols = 1 + len(display_cols)  # method + metrics
    col_spec = "l " + " ".join(["c"] * len(display_cols))

    lines: List[str] = []
    lines.append(f"\\begin{{table}}[{position}]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    if resize:
        lines.append("\\resizebox{\\columnwidth}{!}{%")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header_row = "Method & " + " & ".join(col_headers) + " \\\\"
    lines.append(header_row)
    lines.append("\\midrule")

    # Data rows
    for i, (_, row) in enumerate(df.iterrows()):
        method = row[method_col] if method_col in df.columns else str(row.name)
        method_display = display_name(method)

        # Optionally bold the method name
        if highlight_method and method == highlight_method:
            method_display = f"\\textbf{{{method_display}}}"

        method_display = _latex_escape(method_display)
        cells_str = " & ".join(formatted_cells[i])
        lines.append(f"{method_display} & {cells_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    if resize:
        lines.append("}")  # close \resizebox

    lines.append("\\end{table}")

    return "\n".join(lines)


def save_latex_table(
    tex_str: str,
    path: Union[str, Path],
    *,
    standalone_header: bool = False,
) -> Path:
    """Save a LaTeX table string to a ``.tex`` file.

    Parameters
    ----------
    tex_str : str
        LaTeX source (as returned by ``to_booktabs``).
    path : str or Path
        Output file path (should end in ``.tex``).
    standalone_header : bool
        If ``True``, prepend ``\\documentclass`` / ``\\usepackage``
        so the file is directly compilable with ``pdflatex``.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = tex_str
    if standalone_header:
        preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{graphicx}\n"
            "\\begin{document}\n\n"
        )
        postamble = "\n\n\\end{document}\n"
        content = preamble + content + postamble

    path.write_text(content, encoding="utf-8")
    logger.info("Saved LaTeX table: %s", path)
    return path.resolve()


def build_results_table(
    agg_data: Dict[str, Dict[str, Tuple[float, float]]],
    metrics: Sequence[str],
    column_names: Dict[str, str],
    column_directions: Dict[str, MetricDir],
    caption: str,
    label: str,
    *,
    precision: int = 3,
    highlight_method: Optional[str] = None,
    delta_m: Optional[Dict[str, float]] = None,
) -> str:
    """Build a results table from raw aggregated data.

    This is a higher-level wrapper that converts the output of
    ``metrics.aggregate_over_seeds`` into a formatted LaTeX table.

    Parameters
    ----------
    agg_data : dict[str, dict[str, tuple[float, float]]]
        ``method → metric → (mean, std)`` mapping.
    metrics : sequence of str
        Metrics to include, in display order.
    column_names : dict[str, str]
        ``metric → LaTeX header`` mapping.
    column_directions : dict[str, MetricDir]
        ``metric → direction`` for highlighting.
    caption, label : str
        LaTeX caption and label.
    precision : int
        Decimal precision.
    highlight_method : str, optional
        Method whose name to bold.
    delta_m : dict[str, float], optional
        ``method → ΔM`` values to append as an extra column.

    Returns
    -------
    str
        Complete LaTeX table source.
    """
    # Build DataFrame
    records = []
    for method, metric_data in agg_data.items():
        record: Dict[str, Any] = {"method": method}
        for m in metrics:
            mean, std = metric_data.get(m, (np.nan, np.nan))
            record[f"{m}_mean"] = mean
            record[f"{m}_std"] = std
        if delta_m and method in delta_m:
            record["delta_m_mean"] = delta_m[method]
            record["delta_m_std"] = 0.0  # ΔM computed from means
        records.append(record)

    df = pd.DataFrame(records)

    # Include ΔM in the metrics list if provided
    display_metrics = list(metrics)
    col_names = dict(column_names)
    col_dirs = dict(column_directions)

    if delta_m:
        display_metrics.append("delta_m")
        col_names["delta_m"] = r"$\Delta_M$ $\uparrow$"
        col_dirs["delta_m"] = MetricDir.MAX

    return to_booktabs(
        df,
        caption=caption,
        label=label,
        column_names=col_names,
        column_directions=col_dirs,
        precision=precision,
        mean_std_columns=display_metrics,
        highlight_method=highlight_method,
    )
