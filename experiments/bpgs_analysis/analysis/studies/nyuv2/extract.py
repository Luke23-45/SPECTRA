"""
NYUv2 benchmark data extraction pipeline.

Loads all NYUv2 benchmark runs (4 methods × 3 seeds), parses metrics,
computes ΔM, and produces structured data for tables and figures.

Data source
-----------
``nyuv2/final_output_versions/{method_path}/seed_{s}/csv_logs/*/metrics.csv``

CRITICAL path nesting:
- **Double-nested**: ``bpgs/bpgs/seed_42/``, ``static/static/seed_42/``
- **Single-nested**: ``kendall/seed_42/``, ``uwso/seed_42/``

All runs train for 120 epochs with no early stopping.
Selection metric is ``val/miou`` (mode=max), but we report
**final-epoch** metrics for consistency (not best-checkpoint).

CHECKPOINT NOTE: Some methods (e.g. kendall) have non-null checkpoint
data while others (bpgs, static) have null checkpoints.  We always
use final-epoch metrics for fair comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.common.constants import (
    NYUV2_METHODS,
    NYUV2_TABLE_METRICS,
    NYUV2_TASKS,
    SEEDS,
    MetricDir,
    display_name,
)
from analysis.common.io import (
    DiscoveryReport,
    discover_nyuv2_runs,
    load_all_training_metrics,
    load_run_summary,
    resolve_data_root,
)
from analysis.common.metrics import (
    aggregate_over_seeds,
    build_aggregated_table,
    compute_delta_m_for_nyuv2,
    extract_final_epoch,
)

logger = logging.getLogger(__name__)


def extract_nyuv2_data(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = NYUV2_METHODS,
    seeds: Sequence[int] = SEEDS,
    baseline_method: str = "static",
) -> Dict[str, Any]:
    """Extract all NYUv2 benchmark data.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str
        Benchmark method names.
    seeds : sequence of int
        Seed values.
    baseline_method : str
        Baseline for ΔM computation (default: ``"static"``).

    Returns
    -------
    dict
        Keys:

        - ``"report"``: ``DiscoveryReport``.
        - ``"epoch_metrics"``: ``method → seed → DataFrame``.
        - ``"final_epoch"``: ``method → seed → Series``.
        - ``"aggregated"``: ``method → metric → (mean, std)``.
        - ``"summary_df"``: Aggregated summary DataFrame.
        - ``"delta_m"``: ``method → ΔM value`` (float).
    """
    if data_root is None:
        data_root = resolve_data_root()

    # Discover runs
    report = discover_nyuv2_runs(data_root, methods, seeds)
    report.warn_if_incomplete("nyuv2")

    # Load epoch-level metrics
    epoch_metrics = load_all_training_metrics(report)

    # Extract final-epoch values
    final_epoch: Dict[str, Dict[int, pd.Series]] = {}
    for method, seed_data in epoch_metrics.items():
        final_epoch[method] = {}
        for seed, df in seed_data.items():
            try:
                final_epoch[method][seed] = extract_final_epoch(df)
            except ValueError as exc:
                logger.warning(
                    "Could not extract final epoch for %s seed=%d: %s",
                    method, seed, exc,
                )

    # Aggregate across seeds
    target_metrics = [m[0] for m in NYUV2_TABLE_METRICS]
    aggregated: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for method, seed_series in final_epoch.items():
        if seed_series:
            aggregated[method] = aggregate_over_seeds(
                seed_series, target_metrics
            )

    # Build summary DataFrame
    summary_df = build_aggregated_table(
        epoch_metrics,
        target_metrics,
        extraction="final",
    )

    # Compute ΔM for each method
    delta_m = _compute_all_delta_m(
        aggregated, baseline_method, target_metrics
    )

    return {
        "report": report,
        "epoch_metrics": epoch_metrics,
        "final_epoch": final_epoch,
        "aggregated": aggregated,
        "summary_df": summary_df,
        "delta_m": delta_m,
    }


def _compute_all_delta_m(
    aggregated: Dict[str, Dict[str, Tuple[float, float]]],
    baseline_method: str,
    all_metrics: Sequence[str],
) -> Dict[str, float]:
    """Compute ΔM for all methods against the baseline.

    Uses the canonical 3-task NYUv2 definition:
    mIoU (↑), Abs Rel (↓), Normal Angle (↓).
    """
    delta_m: Dict[str, float] = {}

    if baseline_method not in aggregated:
        logger.warning(
            "Baseline method '%s' not found in aggregated data — "
            "cannot compute ΔM",
            baseline_method,
        )
        return delta_m

    baseline_means = {
        metric: aggregated[baseline_method][metric][0]
        for metric in [t.metric for t in NYUV2_TASKS]
        if metric in aggregated[baseline_method]
    }

    for method, metric_data in aggregated.items():
        method_means = {
            metric: metric_data[metric][0]
            for metric in [t.metric for t in NYUV2_TASKS]
            if metric in metric_data
        }

        if len(method_means) != len(NYUV2_TASKS):
            logger.warning(
                "Incomplete ΔM metrics for method '%s': have %s, need %s",
                method,
                sorted(method_means.keys()),
                [t.metric for t in NYUV2_TASKS],
            )
            delta_m[method] = np.nan
            continue

        try:
            delta_m[method] = compute_delta_m_for_nyuv2(
                method_means, baseline_means
            )
        except Exception as exc:
            logger.error("ΔM computation failed for %s: %s", method, exc)
            delta_m[method] = np.nan

    return delta_m


def extract_nyuv2_training_curves(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = NYUV2_METHODS,
    seeds: Sequence[int] = SEEDS,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Extract training curves for NYUv2 methods.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.
    metrics : sequence of str, optional
        Metrics to extract.  Defaults to ``["val/total_loss",
        "val/miou"]``.

    Returns
    -------
    dict[str, dict[str, dict[str, ndarray]]]
        ``metric → method → {"epochs": ..., "mean": ..., "std": ...}``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    if metrics is None:
        metrics = ["val/total_loss", "val/miou"]

    report = discover_nyuv2_runs(data_root, methods, seeds)
    epoch_data = load_all_training_metrics(report)

    all_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for metric in metrics:
        all_curves[metric] = {}

        for method, seed_data in epoch_data.items():
            if not seed_data:
                continue

            seed_dfs = list(seed_data.values())
            min_epochs = min(len(df) for df in seed_dfs)
            epochs = seed_dfs[0]["epoch"].values[:min_epochs]

            values = []
            for df in seed_dfs:
                if metric in df.columns:
                    vals = df[metric].values[:min_epochs]
                    values.append(vals)

            if values:
                stacked = np.array(values, dtype=np.float64)
                all_curves[metric][method] = {
                    "epochs": epochs,
                    "mean": np.nanmean(stacked, axis=0),
                    "std": np.nanstd(stacked, axis=0, ddof=1),
                }

    return all_curves


def extract_per_task_scores(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = NYUV2_METHODS,
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Extract per-task scores for NYUv2 methods.

    Returns
    -------
    dict[str, dict[str, tuple[float, float]]]
        ``method → task_metric → (mean, std)`` for the canonical
        NYUv2 task metrics.
    """
    if data_root is None:
        data_root = resolve_data_root()

    result = extract_nyuv2_data(data_root, methods, seeds)
    return result["aggregated"]
