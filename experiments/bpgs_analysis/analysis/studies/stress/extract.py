"""
Stress test data extraction pipeline.

Handles three sub-experiments:
1. **Synthetic Scale Stress (02)**: macro_score vs scale factor (x1→x1000).
2. **Pure Loss Rescaling (06)**: Same structure as scale stress.
3. **Heterogeneous Mixed Stress (07)**: macro_score across regimes
   (clean, noisy, conflict).

Data source
-----------
- Scale/Rescaling: ``stress/final_outputs/{exp}/x{scale}/{method}/seed_{s}/metrics/results.csv``
- Heterogeneous:   ``stress/final_outputs/07_.../\\{regime\\}/{method}/seed_{s}/metrics/results.csv``

Unlike training CSVs, stress ``results.csv`` files are dense (one row
per seed) and contain pre-computed macro_score, worst_task_score, etc.

NOTE: Directories like ``exp_03_imbalance_robustness``,
``exp_07_pure_loss_rescaling``, and ``exp_08_heterogeneous_regime``
contain only metadata (``latest_run.json``) and must be IGNORED
during data extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.common.constants import (
    SEEDS,
    STRESS_HETERO_METHODS,
    STRESS_REGIMES,
    STRESS_SCALE_METHODS,
    STRESS_SCALES,
    MetricDir,
    display_name,
)
from analysis.common.io import (
    DiscoveryReport,
    discover_stress_hetero_runs,
    discover_stress_scale_runs,
    load_all_stress_results,
    load_stress_results,
    resolve_data_root,
)
from analysis.common.metrics import (
    aggregate_over_seeds,
    compute_degradation,
)

logger = logging.getLogger(__name__)


def extract_scale_stress(
    data_root: Optional[Path] = None,
    experiment: str = "scale",
    methods: Sequence[str] = STRESS_SCALE_METHODS,
    seeds: Sequence[int] = SEEDS,
    scales: Sequence[int] = STRESS_SCALES,
) -> Dict[str, Any]:
    """Extract scale stress test results.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    experiment : str
        ``"scale"`` for 02_synthetic_scale_stress,
        ``"rescaling"`` for 06_pure_loss_rescaling.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.
    scales : sequence of int
        Scale factors.

    Returns
    -------
    dict
        Keys:

        - ``"reports"``: ``scale → DiscoveryReport``.
        - ``"raw_results"``: ``scale → method → seed → DataFrame``.
        - ``"aggregated"``: ``scale → method → metric → (mean, std)``.
        - ``"scale_curves"``: ``method → {"scales": ..., "mean": ..., "std": ...}``
          for macro_score.
        - ``"degradation"``: ``method → degradation_ratio`` (x1 → x1000).
    """
    if data_root is None:
        data_root = resolve_data_root()

    # Discover all runs across scales
    reports = discover_stress_scale_runs(
        data_root, experiment, methods, seeds, scales
    )

    # Load raw results per scale
    raw_results: Dict[int, Dict[str, Dict[int, pd.DataFrame]]] = {}
    for scale, report in reports.items():
        report.warn_if_incomplete(f"stress_{experiment}_x{scale}")
        raw_results[scale] = load_all_stress_results(report)

    # Aggregate across seeds per scale
    target_metrics = [
        "macro_score", "worst_task_score", "best_task_score", "score_std",
    ]
    aggregated: Dict[int, Dict[str, Dict[str, Tuple[float, float]]]] = {}

    for scale in scales:
        aggregated[scale] = {}
        for method in methods:
            seed_data = raw_results.get(scale, {}).get(method, {})
            if not seed_data:
                continue

            # Build seed → Series for aggregation
            seed_series: Dict[int, pd.Series] = {}
            for seed, df in seed_data.items():
                if not df.empty:
                    seed_series[seed] = df.iloc[0]

            if seed_series:
                aggregated[scale][method] = aggregate_over_seeds(
                    seed_series, target_metrics
                )

    # Build scale curves (macro_score vs scale)
    scale_curves = _build_scale_curves(aggregated, methods, scales)

    # Compute degradation ratios
    degradation = _compute_degradation_ratios(
        aggregated, methods, scales
    )

    return {
        "reports": reports,
        "raw_results": raw_results,
        "aggregated": aggregated,
        "scale_curves": scale_curves,
        "degradation": degradation,
    }


def _build_scale_curves(
    aggregated: Dict[int, Dict[str, Dict[str, Tuple[float, float]]]],
    methods: Sequence[str],
    scales: Sequence[int],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build per-method scale curves for macro_score.

    Returns
    -------
    dict[str, dict[str, ndarray]]
        ``method → {"scales": ..., "mean": ..., "std": ...}``.
    """
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for method in methods:
        means = []
        stds = []
        valid_scales = []

        for scale in scales:
            method_data = aggregated.get(scale, {}).get(method, {})
            if "macro_score" in method_data:
                mean, std = method_data["macro_score"]
                means.append(mean)
                stds.append(std)
                valid_scales.append(scale)

        if valid_scales:
            curves[method] = {
                "scales": np.array(valid_scales),
                "mean": np.array(means),
                "std": np.array(stds),
            }

    return curves


def _compute_degradation_ratios(
    aggregated: Dict[int, Dict[str, Dict[str, Tuple[float, float]]]],
    methods: Sequence[str],
    scales: Sequence[int],
    baseline_scale: int = 1,
    target_scale: int = 1000,
) -> Dict[str, float]:
    """Compute relative macro_score degradation from baseline to target scale.

    Returns
    -------
    dict[str, float]
        ``method → degradation_ratio``.
    """
    degradation: Dict[str, float] = {}

    for method in methods:
        scores_by_scale: Dict[int, float] = {}
        for scale in scales:
            method_data = aggregated.get(scale, {}).get(method, {})
            if "macro_score" in method_data:
                scores_by_scale[scale] = method_data["macro_score"][0]

        degradation[method] = compute_degradation(
            scores_by_scale, baseline_scale, target_scale
        )

    return degradation


def extract_heterogeneous_stress(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = STRESS_HETERO_METHODS,
    seeds: Sequence[int] = SEEDS,
    regimes: Sequence[str] = STRESS_REGIMES,
) -> Dict[str, Any]:
    """Extract heterogeneous mixed stress test results.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.
    regimes : sequence of str
        Regime names.

    Returns
    -------
    dict
        Keys:

        - ``"reports"``: ``regime → DiscoveryReport``.
        - ``"raw_results"``: ``regime → method → seed → DataFrame``.
        - ``"aggregated"``: ``regime → method → metric → (mean, std)``.
        - ``"regime_table"``: ``DataFrame`` with methods as rows,
          regimes as column groups.
    """
    if data_root is None:
        data_root = resolve_data_root()

    # Discover runs
    reports = discover_stress_hetero_runs(
        data_root, methods, seeds, regimes
    )

    # Load raw results per regime
    raw_results: Dict[str, Dict[str, Dict[int, pd.DataFrame]]] = {}
    for regime, report in reports.items():
        report.warn_if_incomplete(f"stress_heterogeneous_{regime}")
        raw_results[regime] = load_all_stress_results(report)

    # Aggregate across seeds per regime
    target_metrics = ["macro_score", "worst_task_score", "score_std"]
    aggregated: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}

    for regime in regimes:
        aggregated[regime] = {}
        for method in methods:
            seed_data = raw_results.get(regime, {}).get(method, {})
            if not seed_data:
                continue

            seed_series: Dict[int, pd.Series] = {}
            for seed, df in seed_data.items():
                if not df.empty:
                    seed_series[seed] = df.iloc[0]

            if seed_series:
                aggregated[regime][method] = aggregate_over_seeds(
                    seed_series, target_metrics
                )

    # Build regime comparison table
    regime_table = _build_regime_table(aggregated, methods, regimes)

    return {
        "reports": reports,
        "raw_results": raw_results,
        "aggregated": aggregated,
        "regime_table": regime_table,
    }


def _build_regime_table(
    aggregated: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    methods: Sequence[str],
    regimes: Sequence[str],
    metric: str = "macro_score",
) -> pd.DataFrame:
    """Build a table of method × regime macro_score values.

    Returns
    -------
    pd.DataFrame
        Index: methods.  Columns: ``(regime, "mean")`` and
        ``(regime, "std")`` pairs.
    """
    records = []
    for method in methods:
        record: Dict[str, Any] = {"method": method}
        for regime in regimes:
            method_data = aggregated.get(regime, {}).get(method, {})
            if metric in method_data:
                mean, std = method_data[metric]
                record[f"{regime}_mean"] = mean
                record[f"{regime}_std"] = std
            else:
                record[f"{regime}_mean"] = np.nan
                record[f"{regime}_std"] = np.nan
        records.append(record)

    return pd.DataFrame(records).set_index("method")


def extract_all_stress(
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extract all stress test data (scale, rescaling, heterogeneous).

    Returns
    -------
    dict
        ``{"scale": {...}, "rescaling": {...}, "heterogeneous": {...}}``.
    """
    if data_root is None:
        data_root = resolve_data_root()

    return {
        "scale": extract_scale_stress(data_root, "scale"),
        "rescaling": extract_scale_stress(data_root, "rescaling"),
        "heterogeneous": extract_heterogeneous_stress(data_root),
    }
