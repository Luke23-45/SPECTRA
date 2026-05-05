"""
Metrics Computation

Five metric groups for comprehensive evaluation:
    - performance: Per-task scores, macro average, worst-task
    - stability: Variance, gradient norms, cosine similarity
    - boundedness: Precision ranges, entropy
    - robustness: Degradation curves, task dominance
    - trajectory: Smoothness, oscillation detection (NEW)
"""

__all__ = []
"""Metric utilities for empirical experiments."""

from spectra.empirical.metrics.boundedness import compute_boundedness_metrics, compute_entropy
from spectra.empirical.metrics.performance import compute_task_performance, summarize_performance
from spectra.empirical.metrics.robustness import compute_robustness_metrics, compute_task_dominance
from spectra.empirical.metrics.stability import compute_stability_metrics
from spectra.empirical.metrics.trajectory import compute_oscillation_metrics, compute_smoothness_metrics

__all__ = [
    "compute_boundedness_metrics",
    "compute_entropy",
    "compute_task_performance",
    "summarize_performance",
    "compute_robustness_metrics",
    "compute_task_dominance",
    "compute_stability_metrics",
    "compute_oscillation_metrics",
    "compute_smoothness_metrics",
]
