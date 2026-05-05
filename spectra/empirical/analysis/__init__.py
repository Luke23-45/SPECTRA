"""
Analysis & Visualization

Publication-ready analysis tools:
    - stability_analysis: Trajectory smoothness
    - trajectory_metrics: Time-series analysis (L2, TV, ZCR)
    - comparative_analysis: Paired comparisons with uncertainty
    - report_generator: LaTeX tables and figures
    - plotters/: Matplotlib-first visualization suite
"""

__all__ = []
"""Analysis helpers for empirical experiments."""

from spectra.empirical.analysis.comparative_analysis import paired_method_deltas, summarize_comparison_frame
from spectra.empirical.analysis.report_generator import collect_results, write_report
from spectra.empirical.analysis.stability_analysis import analyze_stability_histories

__all__ = [
    "paired_method_deltas",
    "summarize_comparison_frame",
    "collect_results",
    "write_report",
    "analyze_stability_histories",
]
