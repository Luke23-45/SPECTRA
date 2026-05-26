"""
Ablation study data extraction pipeline.

Loads all ablation runs (5 BPGS variants + Kendall on NYUv2, 3 seeds each),
parses their sparse metrics CSVs, and produces structured DataFrames
ready for table and figure generation.

Data source
-----------
``abalation/final_outputs/{method}/{prefix}_nyuv2_s{seed}/csv_logs/{prefix}_nyuv2_s{seed}/metrics.csv``

CRITICAL: All BPGS variants use the prefix ``bpgs`` (not the variant name).
Kendall uses the prefix ``kendall``.

All runs train for 60 epochs with no early stopping.  We always
report **final-epoch** metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.common.constants import (
    ABLATION_METHODS,
    NYUV2_TABLE_METRICS,
    SEEDS,
    MetricDir,
    display_name,
)
from analysis.common.io import (
    DiscoveryReport,
    discover_ablation_runs,
    load_all_training_metrics,
    load_metrics_csv,
    load_run_summary,
    resolve_data_root,
)
from analysis.common.metrics import (
    aggregate_over_seeds,
    build_aggregated_table,
    extract_final_epoch,
)

logger = logging.getLogger(__name__)


def extract_ablation_data(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = ABLATION_METHODS,
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, Any]:
    """Extract all ablation study data.

    This is the top-level entry point for the ablation pipeline.
    It discovers runs, loads all metrics, and computes aggregated
    summaries.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.  Auto-detected if None.
    methods : sequence of str
        Ablation method names.
    seeds : sequence of int
        Seed values.

    Returns
    -------
    dict
        Keys:

        - ``"report"``: ``DiscoveryReport`` with run locations.
        - ``"epoch_metrics"``: ``method → seed → DataFrame`` of
          epoch-level metrics.
        - ``"final_epoch"``: ``method → seed → Series`` of
          final-epoch values.
        - ``"aggregated"``: ``method → metric → (mean, std)``
          aggregated across seeds.
        - ``"summary_df"``: ``DataFrame`` with methods as rows,
          metrics as MultiIndex columns (mean/error).
    """
    if data_root is None:
        data_root = resolve_data_root()

    # Discover runs
    report = discover_ablation_runs(data_root, methods, seeds)
    report.warn_if_incomplete("ablation")

    # Load all epoch-level metrics
    epoch_metrics = load_all_training_metrics(report)

    # Extract final-epoch values per run
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

    return {
        "report": report,
        "epoch_metrics": epoch_metrics,
        "final_epoch": final_epoch,
        "aggregated": aggregated,
        "summary_df": summary_df,
    }


def extract_ablation_training_curves(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = ABLATION_METHODS,
    seeds: Sequence[int] = SEEDS,
    metric: str = "val/total_loss",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract training curves for ablation methods.

    Returns per-method arrays of epochs, mean values, and std values
    (aggregated across seeds), suitable for line plotting with
    seed-shaded bands.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.
    metric : str
        Metric to extract for curves.

    Returns
    -------
    dict[str, dict[str, ndarray]]
        ``method → {"epochs": ..., "mean": ..., "std": ...}``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    report = discover_ablation_runs(data_root, methods, seeds)
    epoch_metrics = load_all_training_metrics(report)

    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for method, seed_data in epoch_metrics.items():
        if not seed_data:
            continue

        # Align all seeds to the same epoch range
        seed_dfs = list(seed_data.values())
        min_epochs = min(len(df) for df in seed_dfs)
        epochs = seed_dfs[0]["epoch"].values[:min_epochs]

        # Stack metric values across seeds
        values = []
        for df in seed_dfs:
            col_vals = df[metric].values[:min_epochs]
            values.append(col_vals)

        stacked = np.array(values, dtype=np.float64)

        curves[method] = {
            "epochs": epochs,
            "mean": np.nanmean(stacked, axis=0),
            "std": np.nanstd(stacked, axis=0, ddof=1),
        }

    return curves


def extract_weight_dynamics(
    data_root: Optional[Path] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = SEEDS,
    n_tasks: int = 3,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract BPGS task-weight dynamics over training.

    Returns per-method arrays of epochs and per-task weight values,
    averaged across seeds.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str, optional
        BPGS variant names (defaults to all except kendall).
    seeds : sequence of int
        Seed values.
    n_tasks : int
        Number of tasks (NYUv2 has 3: seg, depth, normals).

    Returns
    -------
    dict[str, dict[str, ndarray]]
        ``method → {"epochs": ..., "weight_0_mean": ..., ...}``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    if methods is None:
        methods = [m for m in ABLATION_METHODS if m != "kendall"]

    report = discover_ablation_runs(data_root, methods, seeds)
    epoch_metrics = load_all_training_metrics(report)

    dynamics: Dict[str, Dict[str, np.ndarray]] = {}

    for method, seed_data in epoch_metrics.items():
        if not seed_data:
            continue

        seed_dfs = list(seed_data.values())
        min_epochs = min(len(df) for df in seed_dfs)
        epochs = seed_dfs[0]["epoch"].values[:min_epochs]

        result: Dict[str, np.ndarray] = {"epochs": epochs}

        for task_idx in range(n_tasks):
            weight_col = f"train/bpgs/weight_{task_idx}"

            task_values = []
            for df in seed_dfs:
                if weight_col in df.columns:
                    vals = df[weight_col].values[:min_epochs]
                    task_values.append(vals)

            if task_values:
                stacked = np.array(task_values, dtype=np.float64)
                result[f"weight_{task_idx}_mean"] = np.nanmean(stacked, axis=0)
                result[f"weight_{task_idx}_std"] = np.nanstd(
                    stacked, axis=0, ddof=1
                )

        dynamics[method] = result

    return dynamics


def extract_theta_dynamics(
    data_root: Optional[Path] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = SEEDS,
    n_tasks: int = 3,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract BPGS theta parameter dynamics over training.

    Returns per-method arrays of epochs and per-task theta values,
    averaged across seeds.  Theta is the raw gradient scaling signal
    before the softmax transformation (unlike weights which are the
    softmax output).

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str, optional
        BPGS variant names (defaults to all except kendall).
    seeds : sequence of int
        Seed values.
    n_tasks : int
        Number of tasks (NYUv2 has 3: seg, depth, normals).

    Returns
    -------
    dict[str, dict[str, ndarray]]
        ``method → {"epochs": ..., "theta_0_mean": ..., ...}``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    if methods is None:
        methods = [m for m in ABLATION_METHODS if m != "kendall"]

    report = discover_ablation_runs(data_root, methods, seeds)
    epoch_metrics = load_all_training_metrics(report)

    dynamics: Dict[str, Dict[str, np.ndarray]] = {}

    for method, seed_data in epoch_metrics.items():
        if not seed_data:
            continue

        seed_dfs = list(seed_data.values())
        min_epochs = min(len(df) for df in seed_dfs)
        epochs = seed_dfs[0]["epoch"].values[:min_epochs]

        result: Dict[str, np.ndarray] = {"epochs": epochs}

        for task_idx in range(n_tasks):
            theta_col = f"train/bpgs/theta_{task_idx}"

            task_values = []
            for df in seed_dfs:
                if theta_col in df.columns:
                    vals = df[theta_col].values[:min_epochs]
                    task_values.append(vals)

            if task_values:
                stacked = np.array(task_values, dtype=np.float64)
                result[f"theta_{task_idx}_mean"] = np.nanmean(stacked, axis=0)
                result[f"theta_{task_idx}_std"] = np.nanstd(
                    stacked, axis=0, ddof=1
                )

        dynamics[method] = result

    return dynamics
