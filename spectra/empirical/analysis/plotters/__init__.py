"""
Plotting Utilities

Matplotlib-first visualization (seaborn optional):
    - trajectory_plots: Training curve visualizations
    - heatmaps: Correlation and metric heatmaps
    - boxplots: Distribution comparisons
"""

__all__ = []
"""Plotting helpers for empirical analyses."""

from spectra.empirical.analysis.plotters.boxplots import plot_metric_boxplot
from spectra.empirical.analysis.plotters.heatmaps import plot_metric_heatmap
from spectra.empirical.analysis.plotters.trajectory_plots import plot_loss_trajectory

__all__ = ["plot_metric_boxplot", "plot_metric_heatmap", "plot_loss_trajectory"]
