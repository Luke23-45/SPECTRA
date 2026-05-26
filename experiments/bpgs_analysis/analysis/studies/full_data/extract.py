"""
Full-data regime data extraction pipeline.

Loads Yeast (03_yeast_regime_check) and RF1 (08_rf1_regime_check)
experiment runs — 6 methods × 3 seeds each.

Data source
-----------
``full_data/final_results/{experiment}/{method}/seed_{s}/csv_logs/*/metrics.csv``

Consistent path pattern. Methods: bpgs, gradnorm_proxy, kendall,
pcgrad, static, uwso.

Important distinction:
- ``val/total_loss`` is the training/early-stopping selection metric.
- Paper-facing reporting should use task metrics, not just loss.
- Yeast primary metrics: macro-F1, micro-F1, hamming accuracy.
- RF1 primary metrics: R², RMSE, MAE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.common.constants import (
    FULL_DATA_METHODS,
    SEEDS,
    MetricDir,
)
from analysis.common.io import (
    DiscoveryReport,
    discover_full_data_runs,
    load_all_training_metrics,
    load_run_summary,
    resolve_data_root,
)
from analysis.common.metrics import (
    aggregate_over_seeds,
    build_aggregated_table,
    extract_final_epoch,
)

logger = logging.getLogger(__name__)


YEAST_TABLE_METRICS: Tuple[Tuple[str, str, MetricDir], ...] = (
    ("val/macro_f1",     r"Macro-F1 $\uparrow$",       MetricDir.MAX),
    ("val/micro_f1",     r"Micro-F1 $\uparrow$",       MetricDir.MAX),
    ("val/hamming_acc",  r"Hamming Acc $\uparrow$",    MetricDir.MAX),
)
"""Yeast primary table metrics."""

RF1_TABLE_METRICS: Tuple[Tuple[str, str, MetricDir], ...] = (
    ("val/r2",          r"$R^2$ $\uparrow$",         MetricDir.MAX),
    ("val/rmse",        r"RMSE $\downarrow$",        MetricDir.MIN),
    ("val/mae",         r"MAE $\downarrow$",         MetricDir.MIN),
)


def extract_full_data(
    data_root: Optional[Path] = None,
    experiment: str = "yeast",
    methods: Sequence[str] = FULL_DATA_METHODS,
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, Any]:
    """Extract full-data regime results for a single dataset.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    experiment : str
        ``"yeast"`` or ``"rf1"``.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.

    Returns
    -------
    dict
        Keys:

        - ``"report"``: ``DiscoveryReport``.
        - ``"epoch_metrics"``: ``method → seed → DataFrame``.
        - ``"final_epoch"``: ``method → seed → Series``.
        - ``"aggregated"``: ``method → metric → (mean, std)``.
        - ``"summary_df"``: Aggregated summary DataFrame.
        - ``"stopped_epochs"``: ``method → seed → int``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    # Discover runs
    report = discover_full_data_runs(data_root, experiment, methods, seeds)
    report.warn_if_incomplete(f"full_data/{experiment}")

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

    # Get stopped epochs from run summaries
    stopped_epochs = _extract_stopped_epochs(report)

    # Determine target metrics based on experiment
    if experiment == "yeast":
        metric_defs = YEAST_TABLE_METRICS
    else:
        metric_defs = RF1_TABLE_METRICS
    target_metrics = [m[0] for m in metric_defs]

    # Aggregate across seeds
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

    return {
        "report": report,
        "epoch_metrics": epoch_metrics,
        "final_epoch": final_epoch,
        "aggregated": aggregated,
        "summary_df": summary_df,
        "stopped_epochs": stopped_epochs,
    }


def _extract_stopped_epochs(
    report: DiscoveryReport,
) -> Dict[str, Dict[int, int]]:
    """Extract stopped_epoch from run_summary.json for all runs.

    Returns
    -------
    dict[str, dict[int, int]]
        ``method → seed → stopped_epoch``.
    """
    stopped: Dict[str, Dict[int, int]] = {}

    for method, seed_map in report.runs.items():
        stopped[method] = {}
        for seed, info in seed_map.items():
            if not info.exists:
                continue
            try:
                summary = load_run_summary(info.path)
                stopped[method][seed] = int(summary.get("stopped_epoch", -1))
            except Exception as exc:
                logger.warning(
                    "Could not read run_summary for %s seed=%d: %s",
                    method, seed, exc,
                )

    return stopped


def extract_both_datasets(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = FULL_DATA_METHODS,
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, Dict[str, Any]]:
    """Extract data for both Yeast and RF1.

    Returns
    -------
    dict[str, dict]
        ``{"yeast": {...}, "rf1": {...}}``, each containing the same
        keys as ``extract_full_data``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    return {
        "yeast": extract_full_data(data_root, "yeast", methods, seeds),
        "rf1": extract_full_data(data_root, "rf1", methods, seeds),
    }


def extract_full_data_training_curves(
    data_root: Optional[Path] = None,
    experiment: str = "yeast",
    methods: Sequence[str] = FULL_DATA_METHODS,
    seeds: Sequence[int] = SEEDS,
    metric: str = "val/total_loss",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract training curves for a full-data experiment.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    experiment : str
        ``"yeast"`` or ``"rf1"``.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.
    metric : str
        Metric to plot.

    Returns
    -------
    dict[str, dict[str, ndarray]]
        ``method → {"epochs": ..., "mean": ..., "std": ...}``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    report = discover_full_data_runs(data_root, experiment, methods, seeds)
    epoch_data = load_all_training_metrics(report)

    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for method, seed_data in epoch_data.items():
        if not seed_data:
            continue

        seed_dfs = list(seed_data.values())
        # Each seed may have different stopped_epoch → align to minimum
        min_epochs = min(len(df) for df in seed_dfs)
        epochs = seed_dfs[0]["epoch"].values[:min_epochs]

        values = []
        for df in seed_dfs:
            if metric in df.columns:
                vals = df[metric].values[:min_epochs]
                values.append(vals)

        if values:
            stacked = np.array(values, dtype=np.float64)
            curves[method] = {
                "epochs": epochs,
                "mean": np.nanmean(stacked, axis=0),
                "std": np.nanstd(stacked, axis=0, ddof=1),
            }

    return curves


def extract_full_data_training_curves_multi(
    data_root: Optional[Path] = None,
    experiment: str = "yeast",
    methods: Sequence[str] = FULL_DATA_METHODS,
    seeds: Sequence[int] = SEEDS,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Extract training curves for multiple metrics (full-data experiment).

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    experiment : str
        ``"yeast"`` or ``"rf1"``.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.
    metrics : sequence of str, optional
        Metrics to extract.  Defaults to ``["val/total_loss"]``.

    Returns
    -------
    dict[str, dict[str, dict[str, ndarray]]]
        ``metric → method → {"epochs": ..., "mean": ..., "std": ...}``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    if metrics is None:
        metrics = ["val/total_loss"]

    report = discover_full_data_runs(data_root, experiment, methods, seeds)
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
