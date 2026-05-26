"""
Stress test figures — curated, publication-grade (v2 — redesigned).

Produces 5 figures:
1. ``stress_robustness_summary``:  3x2 panel (metric x experiment) with panel labels and seed bands.
2. ``scale_stress_curves``:       1x3 panel: metric vs scale factor with +/-1sigma shaded bands.
3. ``degradation_comparison``:    1x3 panel: % degradation per metric (scale vs rescaling grouped bars).
4. ``rescaling_stress_curves``:   1x3 panel: metric vs scale factor with +/-1sigma shaded bands.
5. ``heterogeneous_panel``:       1x3 panel: metric by regime with grouped bars and error bars.

Design rationale
----------------
* **Dumbbell chart removed**: only showed x1 vs x1000, ignoring intermediate
  scales (x10, x100).  Replaced by line+band curves that show the full
  degradation trajectory across all scale factors.
* **Diverging bar removed**: hatch + alpha encoding for metrics was
  illegible.  Replaced by clean line+band curves for rescaling.
* **Single-metric degradation bar removed**: only showed macro_score.
  Replaced by per-metric panels comparing scale vs rescaling.
* **Heterogeneous panel expanded**: now includes score_std (1x3 instead
  of 1x2) with panel labels.
* All figures use panel labels (a), (b), (c) for cross-referencing.
* Legends are deduplicated, properly positioned, and never overlap data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from analysis.common.constants import (
    STRESS_HETERO_METHODS,
    STRESS_REGIMES,
    STRESS_SCALE_METHODS,
    STRESS_SCALES,
    MetricDir,
    display_name,
    get_metric_direction,
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
from analysis.studies.stress.extract import (
    extract_heterogeneous_stress,
    extract_scale_stress,
)

logger = logging.getLogger(__name__)


def _add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.08) -> None:
    """Add a panel label like (a), (b) to an axes."""
    ax.text(
        x, y, f"({label})", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="bottom",
    )


def _build_metric_curves(
    aggregated: dict,
    methods: Sequence[str],
    scales: Sequence[int],
    metric: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build per-method scale curves for a single metric from aggregated data.

    Returns
    -------
    dict[str, dict[str, ndarray]]
        ``method -> {"scales": ..., "mean": ..., "std": ...}``.
    """
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        means, stds, valid_scales = [], [], []
        for scale in scales:
            md = aggregated.get(scale, {}).get(method, {})
            if metric in md:
                m, s = md[metric]
                means.append(m)
                stds.append(s)
                valid_scales.append(scale)
        if valid_scales:
            curves[method] = {
                "scales": np.array(valid_scales),
                "mean": np.array(means),
                "std": np.array(stds),
            }
    return curves


def _compute_degradation_pct(
    aggregated: dict,
    method: str,
    metric: str,
    baseline_scale: int = 1,
    target_scale: int = 1000,
) -> float:
    """Compute % performance degradation for a metric (positive = worse)."""
    md_base = aggregated.get(baseline_scale, {}).get(method, {})
    md_target = aggregated.get(target_scale, {}).get(method, {})
    if metric not in md_base or metric not in md_target:
        return np.nan
    base_val = md_base[metric][0]
    target_val = md_target[metric][0]
    if base_val == 0:
        return np.nan
    # For higher-is-better (MAX): drop = (base - target) / |base| * 100
    # For lower-is-better (MIN):  drop = (target - base) / |base| * 100
    direction = get_metric_direction(metric)
    if direction == MetricDir.MAX:
        return (base_val - target_val) / abs(base_val) * 100
    else:
        return (target_val - base_val) / abs(base_val) * 100


def generate_stress_robustness_summary(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """3x2 panel: (macro, worst-task, score-std) x (scale, rescaling).

    Each subplot shows metric vs scale factor (log x-axis) with
    +/-1sigma shaded bands.  Panel labels (a)-(f) for cross-referencing.
    """
    scale_data = extract_scale_stress(data_root, "scale")
    rescale_data = extract_scale_stress(data_root, "rescaling")
    methods = sort_methods(list(STRESS_SCALE_METHODS))

    rows = [
        ("macro_score",      r"Macro Score $\uparrow$"),
        ("worst_task_score", r"Worst-Task Score $\uparrow$"),
        ("score_std",        r"Score Std $\downarrow$"),
    ]
    cols = [
        (scale_data,   "Scale Stress"),
        (rescale_data, "Loss Rescaling"),
    ]

    panel_idx = 0

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=3, ncols=2, aspect=0.75,
        )

        for row, (metric, ylabel) in enumerate(rows):
            for col, (data, title) in enumerate(cols):
                ax = axes[row, col]
                aggregated = data["aggregated"]
                curves = _build_metric_curves(
                    aggregated, methods, STRESS_SCALES, metric,
                )

                for method in methods:
                    if method not in curves:
                        continue
                    c = curves[method]
                    plot_with_seed_band(
                        ax, c["scales"], c["mean"], c["std"],
                        color=get_color(method),
                        label=display_name(method),
                        linestyle=get_linestyle(method),
                        marker=get_marker(method),
                        markevery=1,
                        linewidth=1.3,
                        alpha_band=0.12,
                    )

                ax.set_xscale("log")
                ax.set_xticks(list(STRESS_SCALES))
                if row == 2:
                    ax.set_xticklabels(
                        [f"\u00d7{s}" for s in STRESS_SCALES], fontsize=6.5,
                    )
                    ax.set_xlabel("Scale Factor")
                else:
                    ax.set_xticklabels([])
                ax.set_xlim(min(STRESS_SCALES), max(STRESS_SCALES))
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=7.5)
                if row == 0:
                    ax.set_title(title, fontsize=8)

                _add_panel_label(ax, chr(97 + panel_idx))
                panel_idx += 1

        # Shared legend centered above all panels
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=3, fontsize=7.5,
            frameon=True, framealpha=0.95,
            edgecolor="#CCC", fancybox=True, handlelength=2.5,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "stress_robustness_summary")


def generate_scale_stress_curves(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """1x3 panel: each metric vs scale factor with +/-1sigma shaded bands.

    Shows the full degradation trajectory from x1 to x1000 across all
    intermediate scales, not just the two endpoints.  Log x-axis for
    scale factor.  Panel labels (a)-(c).
    """
    scale_data = extract_scale_stress(data_root, "scale")
    aggregated = scale_data["aggregated"]
    methods = sort_methods(list(STRESS_SCALE_METHODS))

    metrics = [
        ("macro_score",      r"Macro Score $\uparrow$"),
        ("worst_task_score", r"Worst-Task Score $\uparrow$"),
        ("score_std",        r"Score Std $\downarrow$"),
    ]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.42,
        )

        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[col]
            curves = _build_metric_curves(
                aggregated, methods, STRESS_SCALES, metric,
            )

            for method in methods:
                if method not in curves:
                    continue
                c = curves[method]
                plot_with_seed_band(
                    ax, c["scales"], c["mean"], c["std"],
                    color=get_color(method),
                    label=display_name(method),
                    linestyle=get_linestyle(method),
                    marker=get_marker(method),
                    markevery=1,
                    linewidth=1.4,
                    alpha_band=0.12,
                )

            ax.set_xscale("log")
            ax.set_xticks(list(STRESS_SCALES))
            ax.set_xticklabels(
                [f"\u00d7{s}" for s in STRESS_SCALES], fontsize=6.5,
            )
            ax.set_xlim(min(STRESS_SCALES), max(STRESS_SCALES))
            ax.set_ylabel(ylabel)
            _add_panel_label(ax, chr(97 + col))

        axes[1].set_xlabel("Scale Factor")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.15),
            ncol=3, fontsize=7.5,
            frameon=True, framealpha=0.95,
            edgecolor="#CCC", fancybox=True, handlelength=2.5,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "scale_stress_curves")


def generate_degradation_comparison(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """1x3 panel: % degradation from x1 to x1000 for all 3 metrics.

    Each panel shows one metric with methods on the x-axis.  Two bars
    per method: Scale Stress (solid) and Loss Rescaling (hatched).
    Positive values = performance degradation.  Panel labels (a)-(c).
    """
    scale_data = extract_scale_stress(data_root, "scale")
    rescale_data = extract_scale_stress(data_root, "rescaling")
    methods = sort_methods(list(STRESS_SCALE_METHODS))

    metrics_info = [
        ("macro_score",      r"Macro Score $\uparrow$"),
        ("worst_task_score", r"Worst-Task Score $\uparrow$"),
        ("score_std",        r"Score Std $\downarrow$"),
    ]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.55,
        )

        for col, (metric, panel_title) in enumerate(metrics_info):
            ax = axes[col]
            n_methods = len(methods)
            x = np.arange(n_methods)
            w = 0.32

            scale_degs = [
                _compute_degradation_pct(
                    scale_data["aggregated"], m, metric,
                )
                for m in methods
            ]
            rescale_degs = [
                _compute_degradation_pct(
                    rescale_data["aggregated"], m, metric,
                )
                for m in methods
            ]

            bars_scale = ax.bar(
                x - w / 2, scale_degs, w,
                label="Scale Stress",
                color="#332288", alpha=0.85,
                edgecolor="white", linewidth=0.4,
            )
            bars_rescale = ax.bar(
                x + w / 2, rescale_degs, w,
                label="Loss Rescaling",
                color="#EE7733", alpha=0.85,
                edgecolor="white", linewidth=0.4,
                hatch="//",
            )

            ax.set_xticks(x)
            ax.set_xticklabels([display_name(m) for m in methods])
            ax.set_ylabel("% Degradation")
            ax.set_title(panel_title, fontsize=8)
            ax.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
            ax.margins(y=0.30)

            # Annotate bars
            for bars in [bars_scale, bars_rescale]:
                for bar in bars:
                    h = bar.get_height()
                    if not np.isnan(h) and abs(h) > 0.1:
                        va = "bottom" if h >= 0 else "top"
                        off = 3 if h >= 0 else -3
                        ax.annotate(
                            f"{h:.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, off), textcoords="offset points",
                            ha="center", va=va, fontsize=6,
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                fc="white", ec="none", alpha=0.85,
                            ),
                        )

            _add_panel_label(ax, chr(97 + col))

        # Shared legend centered above all panels
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.15),
            ncol=2, fontsize=7,
            frameon=True, framealpha=0.95,
            edgecolor="#CCC", fancybox=True, handlelength=2.5,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "degradation_comparison")


def generate_rescaling_stress_curves(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """1x3 panel: each metric vs scale factor with +/-1sigma shaded bands.

    Same structure as scale stress curves but for the loss rescaling
    experiment.  Shows the full trajectory across all scale factors.
    Panel labels (a)-(c).
    """
    rescale_data = extract_scale_stress(data_root, "rescaling")
    aggregated = rescale_data["aggregated"]
    methods = sort_methods(list(STRESS_SCALE_METHODS))

    metrics = [
        ("macro_score",      r"Macro Score $\uparrow$"),
        ("worst_task_score", r"Worst-Task Score $\uparrow$"),
        ("score_std",        r"Score Std $\downarrow$"),
    ]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.42,
        )

        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[col]
            curves = _build_metric_curves(
                aggregated, methods, STRESS_SCALES, metric,
            )

            for method in methods:
                if method not in curves:
                    continue
                c = curves[method]
                plot_with_seed_band(
                    ax, c["scales"], c["mean"], c["std"],
                    color=get_color(method),
                    label=display_name(method),
                    linestyle=get_linestyle(method),
                    marker=get_marker(method),
                    markevery=1,
                    linewidth=1.4,
                    alpha_band=0.12,
                )

            ax.set_xscale("log")
            ax.set_xticks(list(STRESS_SCALES))
            ax.set_xticklabels(
                [f"\u00d7{s}" for s in STRESS_SCALES], fontsize=6.5,
            )
            ax.set_xlim(min(STRESS_SCALES), max(STRESS_SCALES))
            ax.set_ylabel(ylabel)
            _add_panel_label(ax, chr(97 + col))

        axes[1].set_xlabel("Scale Factor")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.15),
            ncol=3, fontsize=7.5,
            frameon=True, framealpha=0.95,
            edgecolor="#CCC", fancybox=True, handlelength=2.5,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "rescaling_stress_curves")


def generate_heterogeneous_panel(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """1x3 panel: macro score, worst-task score, and score std by regime.

    Each panel shows one metric with regimes on the x-axis and methods
    as grouped bars with error bars.  Panel labels (a)-(c).
    """
    data = extract_heterogeneous_stress(data_root)
    aggregated = data["aggregated"]
    methods = sort_methods(list(STRESS_HETERO_METHODS))
    regimes = list(STRESS_REGIMES)

    panels = [
        ("macro_score",      r"Macro Score $\uparrow$"),
        ("worst_task_score", r"Worst-Task Score $\uparrow$"),
        ("score_std",        r"Score Std $\downarrow$"),
    ]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.42,
        )

        for col, (metric, ylabel) in enumerate(panels):
            ax = axes[col]
            x = np.arange(len(regimes))
            n = len(methods)
            total_w = 0.72
            bar_w = total_w / n

            for i, method in enumerate(methods):
                offset = (i - n / 2 + 0.5) * bar_w
                means, stds = [], []

                for regime in regimes:
                    md = aggregated.get(regime, {}).get(method, {})
                    if metric in md:
                        m, s = md[metric]
                        means.append(m)
                        stds.append(s)
                    else:
                        means.append(0)
                        stds.append(0)

                bars = ax.bar(
                    x + offset, means, bar_w,
                    label=display_name(method),
                    color=get_color(method),
                    alpha=0.85,
                    edgecolor="white", linewidth=0.4,
                )

                # Make zero-height bars visible
                for bar, val in zip(bars, means):
                    if abs(val) < 1e-9:
                        ax.plot(
                            [bar.get_x(), bar.get_x() + bar.get_width()],
                            [0, 0],
                            color=get_color(method),
                            linewidth=2.0,
                            solid_capstyle="butt",
                            zorder=3,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([r.capitalize() for r in regimes])
            ax.set_ylabel(ylabel)
            ax.margins(y=0.05)
            # Ensure y-axis includes 0
            yl = ax.get_ylim()
            if yl[0] > 0:
                ax.set_ylim(bottom=0)
            _add_panel_label(ax, chr(97 + col))

        axes[1].set_xlabel("Regime")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.15),
            ncol=4, fontsize=7.5,
            frameon=True, framealpha=0.95,
            edgecolor="#CCC", fancybox=True, handlelength=2.5,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "heterogeneous_panel")


def generate_all_stress_figures(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate curated stress figures (5 total)."""
    logger.info("Generating stress test figures...")
    generate_stress_robustness_summary(data_root, output_dir)
    generate_scale_stress_curves(data_root, output_dir)
    generate_degradation_comparison(data_root, output_dir)
    generate_rescaling_stress_curves(data_root, output_dir)
    generate_heterogeneous_panel(data_root, output_dir)
    logger.info("Stress test figures complete.")
