"""
Ablation study figure generation.

Produces training curves, task-weight dynamics, and per-task validation curves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.common.constants import (
    display_name,
    sort_methods,
)
from analysis.common.style import (
    apply_neurips_style,
    create_figure,
    get_color,
    get_linestyle,
    get_marker,
    plot_with_seed_band,
    save_figure,
)
from analysis.studies.ablation.extract import (
    extract_ablation_training_curves,
    extract_theta_dynamics,
    extract_weight_dynamics,
)

logger = logging.getLogger(__name__)


def generate_training_curves(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Training convergence — highlights SL-auto instability."""
    curves = extract_ablation_training_curves(data_root, metric="val/total_loss")
    methods = sort_methods(list(curves.keys()))

    with apply_neurips_style():
        fig, axes = create_figure(width="double", nrows=1, ncols=2, aspect=0.42)
        ax_full, ax_zoom = axes

        for method in methods:
            c = curves[method]
        # Full view
        plot_with_seed_band(
                ax_full, c["epochs"], c["mean"], c["std"],
                color=get_color(method),
                label=display_name(method),
                linestyle=get_linestyle(method),
                marker=get_marker(method),
                markevery=8,
            )
        # Zoom view
        plot_with_seed_band(
                ax_zoom, c["epochs"], c["mean"], c["std"],
                color=get_color(method),
                label=display_name(method),
                linestyle=get_linestyle(method),
                marker=get_marker(method),
                markevery=8,
            )

        ax_full.set_xlabel("Epoch", labelpad=8)
        ax_full.set_ylabel("Validation Loss", labelpad=8)
        ax_full.set_xlim(left=0)
        ax_full.set_title("Full Training Dynamics", pad=12)

        ax_zoom.set_xlabel("Epoch", labelpad=8)
        ax_zoom.set_xlim(20, 60)
        ax_zoom.set_ylim(2.40, 2.95)
        ax_zoom.set_title("Convergence Detail (Epochs 20-60)", pad=12)

        handles, labels = ax_full.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05), # Placed safely above the plot and titles
            ncol=5,
            fontsize=7.5,
            frameon=True,
            framealpha=0.92,
            edgecolor="#CCC",
            fancybox=True,
            handlelength=3.0,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "ablation_training_curves")


def generate_weight_dynamics(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """BPGS task-weight evolution — one panel per variant."""
    dynamics = extract_weight_dynamics(data_root)
    bpgs_methods = [
        m for m in sort_methods(list(dynamics.keys())) 
        if m != "bpgs_stateless_auto"
    ]

    # NYUv2 tasks indexed as weight_0, weight_1, weight_2
    task_info = [
        (0, "Segmentation", "#332288"),
        (1, "Depth",        "#EE7733"),
        (2, "Normals",      "#117733"),
    ]

    with apply_neurips_style():
        n = len(bpgs_methods)
        fig, axes = create_figure(
            width="double", nrows=1, ncols=n, aspect=0.42,
        )
        if n == 1:
            axes = [axes]

        for i, (ax, method) in enumerate(zip(axes, bpgs_methods)):
            dyn = dynamics[method]
            epochs = dyn["epochs"]
            for idx, name, color in task_info:
                mk = f"weight_{idx}_mean"
                sk = f"weight_{idx}_std"
                if mk in dyn:
                    plot_with_seed_band(
                        ax, epochs, dyn[mk], dyn[sk],
                        color=color, label=name,
                        markevery=999, linewidth=1.3,
                        alpha_band=0.10,
                    )
            ax.set_title(display_name(method), fontsize=8, pad=10)
            
            ax.set_ylim(0, 5.2)

        axes[1].set_xlabel("Epoch", labelpad=8)
        axes[0].set_ylabel("Task Weight")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.15),
            ncol=3, fontsize=7.5,
            frameon=True, framealpha=0.92,
            edgecolor="#CCC", fancybox=True,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "ablation_weight_dynamics")


def generate_per_task_val_curves(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Per-task validation metrics — reveals task-level trade-offs."""
    task_specs = [
        ("val/segmentation_miou",  r"Segmentation mIoU $\uparrow$",  "#332288"),
        ("val/depth_abs_rel",      r"Depth Abs Rel $\downarrow$",   "#EE7733"),
        ("val/normals_mean_angle", r"Normal Angle $\downarrow$",    "#117733"),
    ]

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=1, ncols=3, aspect=0.38,
        )

        for ax, (metric, ylabel, _) in zip(axes, task_specs):
            curves = extract_ablation_training_curves(
                data_root, metric=metric,
            )
            methods = sort_methods(list(curves.keys()))

            for method in methods:
                c = curves[method]
                plot_with_seed_band(
                    ax, c["epochs"], c["mean"], c["std"],
                    color=get_color(method),
                    label=display_name(method),
                    linestyle=get_linestyle(method),
                    marker=get_marker(method),
                    markevery=8,
                )

            ax.set_ylabel(ylabel, labelpad=8)
            ax.set_xlim(left=0)
            
            if "depth" in metric:
                ax.set_ylim(0.25, 0.45)
            elif "normals" in metric:
                ax.set_ylim(32.0, 50.0)

        # Apply a single, centered shared X-axis label to the middle plot
        axes[1].set_xlabel("Epoch", labelpad=8)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            fontsize=7.5,
            frameon=True,
            framealpha=0.92,
            edgecolor="#CCC",
            fancybox=True,
            handlelength=3.0,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "ablation_per_task_val_curves")

def generate_all_ablation_figures(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate curated ablation figures (4 total)."""
    logger.info("Generating ablation figures...")
    generate_training_curves(data_root, output_dir)
    generate_weight_dynamics(data_root, output_dir)
    generate_per_task_val_curves(data_root, output_dir)
    logger.info("Ablation figures complete.")
