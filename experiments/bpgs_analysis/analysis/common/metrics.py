"""
Metric extraction and aggregation for the BPGS analysis framework.

Provides functions to:
- Extract best-epoch or final-epoch metrics from per-epoch DataFrames.
- Aggregate metric values across seeds (mean ± std / SEM).
- Compute the ΔM multi-task performance metric.
- Rank methods across datasets.

All functions are designed to work with the DataFrames produced by
``io.load_metrics_csv`` (training runs) and ``io.load_stress_results``
(stress tests).

Design notes
------------
* Aggregation always reports **standard deviation** (not SEM) by default,
  following the plan's guidance that with N=3 seeds we should be honest
  about variance rather than report misleadingly small SEM bars.
* ``compute_delta_m`` follows the standard MTL ΔM formula used in
  MTAN, Auto-Lambda, Nash-MTL, and related works.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from analysis.common.constants import (
    METRIC_DIRECTION,
    NYUV2_TASKS,
    MetricDir,
    get_metric_direction,
)

logger = logging.getLogger(__name__)


def extract_final_epoch(
    df: pd.DataFrame,
    epoch_col: str = "epoch",
) -> pd.Series:
    """Extract the last epoch's metrics from an epoch-level DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Epoch-level metrics (one row per epoch), as returned by
        ``io.load_metrics_csv``.
    epoch_col : str
        Name of the epoch column.

    Returns
    -------
    pd.Series
        All metric values at the final epoch.

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot extract final epoch from empty DataFrame")

    last_row = df.loc[df[epoch_col].idxmax()]
    return last_row


def extract_best_epoch(
    df: pd.DataFrame,
    metric: str,
    direction: Optional[MetricDir] = None,
    epoch_col: str = "epoch",
) -> pd.Series:
    """Extract the epoch with the best value of a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        Epoch-level metrics.
    metric : str
        Column name of the metric to optimize.
    direction : MetricDir, optional
        Whether to maximize or minimize.  If ``None``, looked up from
        the ``METRIC_DIRECTION`` registry.
    epoch_col : str
        Name of the epoch column.

    Returns
    -------
    pd.Series
        All metric values at the best epoch.

    Raises
    ------
    ValueError
        If the metric column is entirely NaN or the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot extract best epoch from empty DataFrame")
    if metric not in df.columns:
        raise KeyError(
            f"Metric '{metric}' not found in DataFrame columns. "
            f"Available: {sorted(df.columns)}"
        )

    if direction is None:
        direction = get_metric_direction(metric)

    # Drop NaN values in the metric column
    valid = df.dropna(subset=[metric])
    if valid.empty:
        raise ValueError(
            f"Metric '{metric}' is entirely NaN — cannot find best epoch"
        )

    if direction == MetricDir.MAX:
        best_idx = valid[metric].idxmax()
    else:
        best_idx = valid[metric].idxmin()

    return df.loc[best_idx]


def aggregate_over_seeds(
    seed_values: Dict[int, pd.Series],
    metrics: Sequence[str],
    *,
    error_type: Literal["std", "sem"] = "std",
) -> Dict[str, Tuple[float, float]]:
    """Aggregate metric values across seeds.

    Parameters
    ----------
    seed_values : dict[int, Series]
        ``seed → Series`` mapping where each Series contains metric
        values for one seed (e.g. from ``extract_final_epoch``).
    metrics : sequence of str
        Column names to aggregate.
    error_type : "std" or "sem"
        Whether to compute standard deviation or standard error
        of the mean.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``metric_name → (mean, error)`` mapping.
    """
    if not seed_values:
        raise ValueError("No seed values provided for aggregation")

    result: Dict[str, Tuple[float, float]] = {}

    for metric in metrics:
        values = []
        for seed, series in sorted(seed_values.items()):
            if metric in series.index and pd.notna(series[metric]):
                values.append(float(series[metric]))
            else:
                logger.warning(
                    "Metric '%s' missing or NaN for seed %d",
                    metric,
                    seed,
                )

        if not values:
            logger.warning("No valid values for metric '%s'", metric)
            result[metric] = (np.nan, np.nan)
            continue

        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))

        if len(arr) > 1:
            if error_type == "std":
                error = float(np.std(arr, ddof=1))
            else:  # sem
                error = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        else:
            error = 0.0

        result[metric] = (mean, error)

    return result


def build_aggregated_table(
    all_metrics: Dict[str, Dict[int, pd.DataFrame]],
    target_metrics: Sequence[str],
    *,
    extraction: Literal["final", "best"] = "final",
    best_metric: Optional[str] = None,
    error_type: Literal["std", "sem"] = "std",
) -> pd.DataFrame:
    """Build a summary table aggregated across seeds.

    This is the workhorse function for generating paper tables.
    It extracts the relevant epoch from each run, aggregates across
    seeds, and returns a DataFrame with ``mean`` and ``error`` columns
    for each metric.

    Parameters
    ----------
    all_metrics : dict[str, dict[int, DataFrame]]
        ``method → seed → epoch-level DataFrame``, as returned by
        ``io.load_all_training_metrics``.
    target_metrics : sequence of str
        Metric column names to include in the table.
    extraction : "final" or "best"
        Whether to use the final epoch or the best epoch for a
        specific metric.
    best_metric : str, optional
        Required when ``extraction="best"``.  The metric to optimize
        for epoch selection.
    error_type : "std" or "sem"
        Type of error bar.

    Returns
    -------
    pd.DataFrame
        Index: method names.
        Columns: ``MultiIndex`` with ``(metric, "mean")`` and
        ``(metric, "error")`` for each metric.
    """
    if extraction == "best" and best_metric is None:
        raise ValueError(
            "best_metric must be specified when extraction='best'"
        )

    rows: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for method, seed_data in all_metrics.items():
        # Extract the relevant epoch for each seed
        seed_series: Dict[int, pd.Series] = {}
        for seed, df in seed_data.items():
            try:
                if extraction == "final":
                    series = extract_final_epoch(df)
                else:
                    series = extract_best_epoch(df, best_metric)
                seed_series[seed] = series
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Could not extract %s epoch for %s seed=%d: %s",
                    extraction,
                    method,
                    seed,
                    exc,
                )

        if not seed_series:
            logger.warning(
                "No valid seeds for method '%s' — skipping", method
            )
            continue

        rows[method] = aggregate_over_seeds(
            seed_series,
            target_metrics,
            error_type=error_type,
        )

    # Build DataFrame with MultiIndex columns
    records = []
    for method, agg in rows.items():
        record: Dict[str, Any] = {"method": method}
        for metric in target_metrics:
            mean, error = agg.get(metric, (np.nan, np.nan))
            record[(metric, "mean")] = mean
            record[(metric, "error")] = error
        records.append(record)

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df = result_df.set_index("method")

    return result_df


def compute_delta_m(
    method_scores: Dict[str, float],
    baseline_scores: Dict[str, float],
    directions: Dict[str, MetricDir],
) -> float:
    """Compute the ΔM relative multi-task improvement.

    The standard MTL ΔM formula (used by MTAN, Auto-Lambda, etc.):

    .. math::

        \\Delta_M = \\frac{1}{T} \\sum_{t=1}^{T}
            (-1)^{\\mathbb{1}[\\text{lower is better}]}
            \\cdot \\frac{M_t - M_t^{\\text{base}}}{M_t^{\\text{base}}}

    Parameters
    ----------
    method_scores : dict[str, float]
        ``task_metric → score`` for the method being evaluated.
    baseline_scores : dict[str, float]
        ``task_metric → score`` for the baseline (typically "static").
    directions : dict[str, MetricDir]
        ``task_metric → direction`` indicating higher/lower is better.

    Returns
    -------
    float
        ΔM value.  Positive means the method outperforms the baseline
        on average.
    """
    if not method_scores:
        raise ValueError("method_scores is empty")

    tasks = sorted(method_scores.keys())
    if set(tasks) != set(baseline_scores.keys()):
        raise ValueError(
            f"method_scores and baseline_scores must have the same keys. "
            f"Got {sorted(method_scores.keys())} vs {sorted(baseline_scores.keys())}"
        )

    deltas = []
    for task in tasks:
        m_t = method_scores[task]
        b_t = baseline_scores[task]

        if b_t == 0:
            logger.warning(
                "Baseline score for '%s' is zero — skipping in ΔM", task
            )
            continue

        direction = directions.get(task, get_metric_direction(task))

        # Sign: +1 if higher is better, −1 if lower is better
        sign = 1.0 if direction == MetricDir.MAX else -1.0
        delta = sign * (m_t - b_t) / abs(b_t)
        deltas.append(delta)

    if not deltas:
        return 0.0

    return float(np.mean(deltas))


def compute_delta_m_for_nyuv2(
    method_scores: Dict[str, float],
    baseline_scores: Dict[str, float],
) -> float:
    """Compute ΔM for NYUv2 using the canonical 3-task definition.

    Tasks: mIoU (↑), Abs Rel (↓), Normal Angle (↓).
    Baseline: ``static`` method.

    Parameters
    ----------
    method_scores, baseline_scores : dict[str, float]
        Must contain keys matching ``NYUV2_TASKS[*].metric``.

    Returns
    -------
    float
        ΔM value.
    """
    directions = {t.metric: t.direction for t in NYUV2_TASKS}
    task_metrics = [t.metric for t in NYUV2_TASKS]

    # Filter to just the ΔM task metrics
    m_filtered = {k: method_scores[k] for k in task_metrics}
    b_filtered = {k: baseline_scores[k] for k in task_metrics}

    return compute_delta_m(m_filtered, b_filtered, directions)


def rank_methods(
    scores: Dict[str, Dict[str, float]],
    directions: Dict[str, MetricDir],
) -> pd.DataFrame:
    """Rank methods per metric and compute average rank.

    Parameters
    ----------
    scores : dict[str, dict[str, float]]
        ``method → metric → value`` mapping.
    directions : dict[str, MetricDir]
        ``metric → direction`` for each metric.

    Returns
    -------
    pd.DataFrame
        Index: method names.  Columns: per-metric ranks + ``avg_rank``.
        Rank 1 = best.
    """
    methods = sorted(scores.keys())
    metrics = sorted(
        {m for method_scores in scores.values() for m in method_scores}
    )

    # Build a scores matrix
    score_df = pd.DataFrame(
        {m: {metric: scores[m].get(metric, np.nan) for metric in metrics}
         for m in methods}
    ).T  # methods as rows

    # Rank per metric
    rank_df = pd.DataFrame(index=methods, columns=metrics, dtype=float)

    for metric in metrics:
        direction = directions.get(metric, get_metric_direction(metric))
        ascending = direction == MetricDir.MIN

        rank_df[metric] = score_df[metric].rank(
            ascending=ascending,
            method="min",
            na_option="bottom",
        )

    rank_df["avg_rank"] = rank_df[metrics].mean(axis=1)
    rank_df = rank_df.sort_values("avg_rank")

    return rank_df


def compute_degradation(
    scores_by_scale: Dict[int, float],
    baseline_scale: int = 1,
    target_scale: int = 1000,
) -> float:
    """Compute relative performance degradation between two scales.

    .. math::

        \\text{degradation} = \\frac{S_{\\text{base}} - S_{\\text{target}}}
                                     {S_{\\text{base}}}

    Parameters
    ----------
    scores_by_scale : dict[int, float]
        ``scale → macro_score`` mapping.
    baseline_scale : int
        Reference scale (numerator baseline).
    target_scale : int
        Target scale (denominator comparison).

    Returns
    -------
    float
        Relative degradation.  Positive means performance decreased.
    """
    s_base = scores_by_scale.get(baseline_scale)
    s_target = scores_by_scale.get(target_scale)

    if s_base is None or s_target is None:
        logger.warning(
            "Cannot compute degradation: missing scale %d or %d",
            baseline_scale,
            target_scale,
        )
        return np.nan

    if s_base == 0:
        return np.nan

    return float((s_base - s_target) / abs(s_base))
