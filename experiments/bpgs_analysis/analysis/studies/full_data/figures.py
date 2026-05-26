"""
Full-data regime figures — training curves comparing methods.

Produces 2 figures:
- ``yeast_training_curves``: Multi-panel training curves for Yeast
  (Total Loss, Macro-F1, Micro-F1, Hamming Acc).
- ``rf1_training_curves``:  Multi-panel training curves for RF1
  (Total Loss, R², RMSE, MAE).

Each figure shows all methods on the same axes with ±1σ seed bands,
making method comparison immediate and honest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.common.constants import (
    FULL_DATA_METHODS,
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
from analysis.studies.full_data.extract import (
    extract_full_data_training_curves_multi,
)

logger = logging.getLogger(__name__)


def generate_yeast_training_curves(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Multi-panel training curves for Yeast validation metrics.

    4 panels: Total Loss, Macro-F1, Micro-F1, Hamming Acc.
    All methods on the same axes with ±1σ seed bands.
    """
    panel_info = [
        ("val/total_loss",   "Total Loss",    "Validation Loss"),
        ("val/macro_f1",     "Macro F1",      "Macro F1"),
        ("val/micro_f1",     "Micro F1",      "Micro F1"),
        ("val/hamming_acc",  "Hamming Acc",   "Hamming Accuracy"),
    ]
    metrics = [p[0] for p in panel_info]
    all_curves = extract_full_data_training_curves_multi(
        data_root, experiment="yeast", metrics=metrics,
    )
    methods = sort_methods(list(FULL_DATA_METHODS))

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=2, ncols=2, aspect=0.55,
        )

        for ax, (metric_key, ylabel, title) in zip(axes.flat, panel_info):
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
                    markevery=5,
                )
            ax.set_ylabel(ylabel)
            ax.set_xlim(left=0)
        
        # Only label the bottom row's x-axis
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 1].set_xlabel("Epoch")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=6,
            fontsize=6.5,
            frameon=True,
            framealpha=0.92,
            edgecolor="#CCC",
            fancybox=True,
            handlelength=1.5,
            columnspacing=0.8,
            handletextpad=0.4,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "yeast_training_curves")


def generate_rf1_training_curves(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Multi-panel training curves for RF1 validation metrics.

    4 panels: Total Loss, R², RMSE, MAE.
    All methods on the same axes with ±1σ seed bands.
    """
    panel_info = [
        ("val/total_loss", "Total Loss", "Validation Loss"),
        ("val/r2",         r"$R^2$",     r"$R^2$"),
        ("val/rmse",       "RMSE",       "RMSE"),
        ("val/mae",        "MAE",        "MAE"),
    ]
    metrics = [p[0] for p in panel_info]
    all_curves = extract_full_data_training_curves_multi(
        data_root, experiment="rf1", metrics=metrics,
    )
    methods = sort_methods(list(FULL_DATA_METHODS))

    with apply_neurips_style():
        fig, axes = create_figure(
            width="double", nrows=2, ncols=2, aspect=0.55,
        )

        for ax, (metric_key, ylabel, title) in zip(axes.flat, panel_info):
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
                    markevery=5,
                )
            ax.set_ylabel(ylabel)
            ax.set_xlim(left=0)

        # Only label the bottom row's x-axis
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 1].set_xlabel("Epoch")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=6,
            fontsize=6.5,
            frameon=True,
            framealpha=0.92,
            edgecolor="#CCC",
            fancybox=True,
            handlelength=1.5,
            columnspacing=0.8,
            handletextpad=0.4,
        )

        if output_dir:
            save_figure(fig, Path(output_dir) / "rf1_training_curves")


def generate_all_full_data_figures(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate full-data training curve figures."""
    logger.info("Generating full-data training curves...")
    generate_yeast_training_curves(data_root, output_dir)
    generate_rf1_training_curves(data_root, output_dir)
    logger.info("Full-data figures complete.")
