"""
NYUv2 figure generation.

Produces task training curves, performance comparisons, relative-improvement
plots, Delta-M summaries, and metric heatmaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.common.constants import (
    NYUV2_METHODS,
    NYUV2_TABLE_METRICS,
    NYUV2_TASKS,
    SEEDS,
    MetricDir,
    display_name,
    sort_methods,
)
from analysis.common.style import (
    add_shared_legend_below,
    apply_neurips_style,
    create_figure,
    get_color,
    get_linestyle,
    get_marker,
    plot_with_seed_band,
    save_figure,
)
from analysis.studies.nyuv2.extract import (
    extract_nyuv2_data,
    extract_nyuv2_training_curves,
)

logger = logging.getLogger(__name__)


def _add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.08) -> None:
    """Add a panel label like (a), (b) to an axes."""
    ax.text(
        x, y, f"({label})", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="bottom",
    )


def _extract_per_seed(
    data_root: Optional[Path] = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return method -> seed -> {metric: value} for NYUv2."""
    data = extract_nyuv2_data(data_root)
    result: Dict[str, Dict[int, Dict[str, float]]] = {}
    for method, seed_series in data["final_epoch"].items():
        result[method] = {}
        for seed, series in seed_series.items():
            result[method][seed] = {
                m: float(series[m]) for m in series.index
                if not np.isnan(series[m])
            }
    return result


def _compute_aggregated(
    per_seed: Dict[str, Dict[int, Dict[str, float]]],
    methods: Sequence[str],
    metrics: Sequence[str],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute mean +/- std per method per metric from per-seed data."""
    agg: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for method in methods:
        agg[method] = {}
        for metric in metrics:
            vals = [
                per_seed.get(method, {}).get(s, {}).get(metric, np.nan)
                for s in SEEDS
            ]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                agg[method][metric] = (
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1)),
                )
            else:
                agg[method][metric] = (np.nan, np.nan)
    return agg


def _compute_relative_improvement(
    per_seed: Dict[str, Dict[int, Dict[str, float]]],
    agg: Dict[str, Dict[str, Tuple[float, float]]],
    methods: Sequence[str],
    metrics_info: Sequence[Tuple[str, str, MetricDir]],
    baseline_method: str = "static",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute % improvement over baseline per method per metric.

    Returns
    -------
    dict[str, dict[str, tuple[float, float]]]
        ``method -> metric -> (pct_improvement, std)``.
        Positive = method outperforms baseline.
    """
    metric_keys = [m[0] for m in metrics_info]
    rel: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for method in methods:
        if method == baseline_method:
            continue
        rel[method] = {}
        for i, metric in enumerate(metric_keys):
            base_mean = agg[baseline_method][metric][0]
            method_mean = agg[method][metric][0]
            if np.isnan(base_mean) or base_mean == 0:
                rel[method][metric] = (np.nan, np.nan)
                continue

            direction = metrics_info[i][2]
            sign = 1.0 if direction == MetricDir.MAX else -1.0
            pct = sign * (method_mean - base_mean) / abs(base_mean) * 100

            # Propagate std via per-seed deltas
            base_vals = [
                per_seed.get(baseline_method, {}).get(s, {}).get(metric, np.nan)
                for s in SEEDS
            ]
            method_vals = [
                per_seed.get(method, {}).get(s, {}).get(metric, np.nan)
                for s in SEEDS
            ]
            seed_deltas = []
            for bv, mv in zip(base_vals, method_vals):
                if not np.isnan(bv) and not np.isnan(mv) and bv != 0:
                    seed_deltas.append(sign * (mv - bv) / abs(bv) * 100)
            delta_std = (
                float(np.std(seed_deltas, ddof=1))
                if len(seed_deltas) > 1
                else 0.0
            )
            rel[method][metric] = (pct, delta_std)

    return rel


def generate_task_training_curves(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """1x3 panel training curves for the 3 canonical task metrics.

    Standard convergence dynamics figure with +/-1sigma shaded bands.
    Panel labels (a)-(c).
    """
    target_metrics = [
        "val/segmentation_miou",
        "val/depth_abs_rel",
        "val/normals_mean_angle",
    ]
    all_curves = extract_nyuv2_training_curves(
        data_root, metrics=target_metrics,
    )
    methods = sort_methods(list(NYUV2_METHODS))

    panel_info = [
        ("val/segmentation_miou", "mIoU", "Segmentation"),
        ("val/depth_abs_rel", "Abs Rel", "Depth"),
        ("val/normals_mean_angle", "Angle (\u00b0)", "Surface Normals"),
    ]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.42,
        )

        for col, (metric_key, ylabel, title) in enumerate(panel_info):
            ax = axes[col]
            curves = all_curves.get(metric_key, {})
            for method in methods:
                if method not in curves:
                    continue
                c = curves[method]
                plot_with_seed_band(
                    ax, c["epochs"], c["mean"], c["std"],
                    color=get_color(method),
                    label=display_name(method),
                    linestyle=get_linestyle(method),
                    marker=get_marker(method),
                    markevery=15,
                )
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=8)
            ax.set_xlim(left=0)
            if col == 1:
                ax.set_ylim(0.20, 0.45)
            _add_panel_label(ax, chr(97 + col))

        axes[1].set_xlabel("Epoch")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.15),
            ncol=4, fontsize=7.5,
            frameon=True, framealpha=0.92,
            edgecolor="#CCC", fancybox=True, handlelength=2.5,
            columnspacing=1.0, handletextpad=0.5,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "nyuv2_task_training_curves")


def generate_performance_comparison(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """1x3 grouped bar chart: final performance per task with error bars.

    The standard figure type in MTL papers (MTAN, Nash-MTL, CAGrad).
    Each panel shows one task metric with methods as grouped bars.
    Error bars show +/-1sigma across seeds.  Panel labels (a)-(c).
    """
    per_seed = _extract_per_seed(data_root)
    methods = sort_methods(list(NYUV2_METHODS))

    task_specs = [
        ("val/segmentation_miou", r"mIoU $\uparrow$", "Segmentation"),
        ("val/depth_abs_rel", r"Abs Rel $\downarrow$", "Depth"),
        ("val/normals_mean_angle", r"Angle $\downarrow$", "Surface Normals"),
    ]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.55,
        )

        for col, (metric, ylabel, title) in enumerate(task_specs):
            ax = axes[col]
            n = len(methods)
            x = np.arange(n)
            bar_w = 0.6

            for i, method in enumerate(methods):
                seed_vals = []
                for seed in SEEDS:
                    val = per_seed.get(method, {}).get(seed, {}).get(metric, np.nan)
                    if not np.isnan(val):
                        seed_vals.append(val)

                if not seed_vals:
                    continue

                mean_val = float(np.mean(seed_vals))
                std_val = float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0

                bars = ax.bar(
                    i, mean_val, bar_w,
                    color=get_color(method),
                    label=display_name(method),
                    alpha=0.85,
                    edgecolor="white", linewidth=0.4,
                )

                for bar in bars:
                    h = bar.get_height()
                    if not np.isnan(h):
                        if abs(h) < 0.01:
                            text = f"{h:.4f}"
                        elif abs(h) < 1:
                            text = f"{h:.3f}"
                        else:
                            text = f"{h:.2f}"
                            
                        va = "bottom" if h >= 0 else "top"
                        off = 3 if h >= 0 else -3
                        ax.annotate(
                            text,
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, off), textcoords="offset points",
                            ha="center", va=va, fontsize=6,
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                fc="white", ec="none", alpha=0.85,
                            ),
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([display_name(m) for m in methods])
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=8)
            ax.margins(y=0.15)
            _add_panel_label(ax, chr(97 + col))

        if output_dir:
            save_figure(fig, Path(output_dir) / "nyuv2_performance_comparison")


def generate_relative_improvement(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    baseline_method: str = "static",
) -> None:
    """1x3 bar chart: % improvement over baseline per task.

    Standard figure in Nash-MTL, CAGrad, GradNorm papers.  Each panel
    shows one task with methods as bars.  Positive = improvement over
    baseline; negative = degradation.  Error bars from per-seed deltas.
    Panel labels (a)-(c).
    """
    per_seed = _extract_per_seed(data_root)
    methods = sort_methods(list(NYUV2_METHODS))
    metrics_info = NYUV2_TABLE_METRICS

    # Use only the 3 canonical task metrics for the panels
    task_metrics_info = [
        ("val/segmentation_miou", r"mIoU $\uparrow$", MetricDir.MAX, "Segmentation"),
        ("val/depth_abs_rel", r"Abs Rel $\downarrow$", MetricDir.MIN, "Depth"),
        ("val/normals_mean_angle", r"Angle $\downarrow$", MetricDir.MIN, "Surface Normals"),
    ]

    agg = _compute_aggregated(
        per_seed, methods, [m[0] for m in task_metrics_info],
    )
    rel = _compute_relative_improvement(
        per_seed, agg, methods, task_metrics_info, baseline_method,
    )

    non_baseline = [m for m in methods if m != baseline_method]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.55,
        )

        for col, (metric, ylabel, direction, title) in enumerate(task_metrics_info):
            ax = axes[col]
            n = len(non_baseline)
            x = np.arange(n)
            bar_w = 0.6

            for i, method in enumerate(non_baseline):
                if metric not in rel.get(method, {}):
                    continue
                pct, pct_std = rel[method][metric]

                if np.isnan(pct):
                    continue

                bars = ax.bar(
                    i, pct, bar_w,
                    color=get_color(method),
                    label=display_name(method),
                    alpha=0.85,
                    edgecolor="white", linewidth=0.4,
                )

                for bar in bars:
                    h = bar.get_height()
                    if not np.isnan(h) and abs(h) > 0.1:
                        va = "bottom" if h >= 0 else "top"
                        off = 3 if h >= 0 else -3
                        ax.annotate(
                            f"{h:+.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, off), textcoords="offset points",
                            ha="center", va=va, fontsize=6,
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                fc="white", ec="none", alpha=0.85,
                            ),
                        )

            ax.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels([display_name(m) for m in non_baseline])
            ax.set_ylabel(f"% Improvement over {display_name(baseline_method)}")
            ax.set_title(title, fontsize=8)
            ax.margins(y=0.20)
            _add_panel_label(ax, chr(97 + col))

        if output_dir:
            save_figure(fig, Path(output_dir) / "nyuv2_relative_improvement")


def generate_delta_m_comparison(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    baseline_method: str = "static",
) -> None:
    """Bar chart of Delta-M (multi-task improvement) per method.

    Delta-M is the standard multi-task performance metric used across
    MTL literature (MTAN, Auto-Lambda, Nash-MTL).  Positive = method
    outperforms baseline on average across tasks.
    """
    data = extract_nyuv2_data(data_root)
    delta_m = data["delta_m"]
    methods = sort_methods(list(NYUV2_METHODS))

    # Exclude baseline (Delta-M = 0 by definition)
    non_baseline = [m for m in methods if m != baseline_method]

    with apply_neurips_style():
        fig, ax = create_figure(width="single", aspect=0.45)

        n = len(non_baseline)
        y_pos = np.arange(n)
        bar_h = 0.55

        for i, method in enumerate(non_baseline):
            dm = delta_m.get(method, np.nan)
            if np.isnan(dm):
                continue
            ax.barh(
                i, dm, height=bar_h,
                color=get_color(method),
                label=display_name(method),
                alpha=0.9,
                edgecolor="white", linewidth=0.5,
            )
            # Annotate value cleanly without white bounding box
            ha = "left" if dm >= 0 else "right"
            off = 4 if dm >= 0 else -4
            ax.annotate(
                f"{dm:+.3f}",
                xy=(dm, i),
                xytext=(off, 0), textcoords="offset points",
                ha=ha, va="center", fontsize=8, fontweight="bold",
                color="#333",
            )

        ax.axvline(x=0, color="#666", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([display_name(m) for m in non_baseline])
        ax.set_xlabel(r"$\Delta_M$ (vs Static)")
        ax.margins(x=0.25)
        
        # Clean up spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.invert_yaxis()

        if output_dir:
            save_figure(fig, Path(output_dir) / "nyuv2_delta_m_comparison")


def generate_performance_heatmap(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Annotated heatmap of all metrics x methods.

    Clean overview figure showing all metric values at a glance.
    Values are annotated directly on cells.  Color intensity encodes
    relative performance (darker = better within each metric column).
    Standard in benchmark and comparison papers.
    """
    data = extract_nyuv2_data(data_root)
    aggregated = data["aggregated"]
    methods = sort_methods(list(NYUV2_METHODS))
    metrics_info = NYUV2_TABLE_METRICS
    metric_keys = [m[0] for m in metrics_info]
    metric_labels = []
    for _, display, _ in metrics_info:
        metric_labels.append(display.strip())

    # Build value matrix (methods x metrics)
    n_methods = len(methods)
    n_metrics = len(metric_keys)
    value_matrix = np.full((n_methods, n_metrics), np.nan)

    for i, method in enumerate(methods):
        for j, metric in enumerate(metric_keys):
            if metric in aggregated.get(method, {}):
                value_matrix[i, j] = aggregated[method][metric][0]

    # Normalize per column for color mapping (1 = best, 0 = worst)
    norm_matrix = np.full_like(value_matrix, np.nan)
    for j in range(n_metrics):
        direction = metrics_info[j][2]
        col_vals = value_matrix[:, j]
        valid = ~np.isnan(col_vals)
        if not valid.any():
            continue
        lo = np.nanmin(col_vals)
        hi = np.nanmax(col_vals)
        rng = hi - lo
        if rng == 0:
            norm_matrix[valid, j] = 0.5
        else:
            normalized = (col_vals[valid] - lo) / rng
            if direction == MetricDir.MIN:
                normalized = 1.0 - normalized  # flip so higher = better
            norm_matrix[valid, j] = normalized

    with apply_neurips_style():
        fig, ax = create_figure(width="double", aspect=0.45)

        # Use a sequential colormap (darker = better)
        cmap = plt.cm.YlGnBu

        im = ax.imshow(
            norm_matrix, cmap=cmap, aspect="auto",
            vmin=0, vmax=1, interpolation="nearest",
        )

        # Annotate cells with raw values
        for i in range(n_methods):
            for j in range(n_metrics):
                val = value_matrix[i, j]
                if np.isnan(val):
                    continue
                # Choose text color for readability
                nv = norm_matrix[i, j]
                txt_color = "white" if nv > 0.55 else "black"
                # Format: 3 significant digits
                if abs(val) < 0.01:
                    text = f"{val:.4f}"
                elif abs(val) < 1:
                    text = f"{val:.3f}"
                else:
                    text = f"{val:.2f}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    fontsize=7, color=txt_color, fontweight="medium",
                )

        ax.set_xticks(np.arange(n_metrics))
        ax.set_xticklabels(metric_labels, fontsize=7)
        ax.set_yticks(np.arange(n_methods))
        ax.set_yticklabels([display_name(m) for m in methods], fontsize=7)
        ax.grid(False)

        # Add "better" arrow indicator per column
        for j in range(n_metrics):
            direction = metrics_info[j][2]
            arrow = r"$\uparrow$" if direction == MetricDir.MAX else r"$\downarrow$"
            # Already in metric label from constants

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Relative Performance", fontsize=7)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Worse", "", "Better"], fontsize=6)

        if output_dir:
            save_figure(fig, Path(output_dir) / "nyuv2_performance_heatmap")


def generate_all_nyuv2_figures(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate curated NYUv2 figures (v3 — 5 total)."""
    logger.info("Generating NYUv2 figures (v3)...")
    generate_task_training_curves(data_root, output_dir)
    generate_performance_comparison(data_root, output_dir)
    generate_relative_improvement(data_root, output_dir)
    generate_delta_m_comparison(data_root, output_dir)
    generate_performance_heatmap(data_root, output_dir)
    logger.info("NYUv2 figures complete.")
