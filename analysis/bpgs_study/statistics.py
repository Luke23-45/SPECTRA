from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


def confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) using the t-distribution."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if n == 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1) / math.sqrt(n))
    t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return (mean, mean - t_crit * se, mean + t_crit * se)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    mean = float(arr.mean())
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return (mean, lo, hi)


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Welch's t-test (unequal variances). Returns t_stat, p_value, df."""
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if len(xa) < 2 or len(xb) < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"), "df": float("nan")}
    result = sp_stats.ttest_ind(xa, xb, equal_var=False)
    return {
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "df": float(result.df),
    }


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples. Returns W, p_value."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[diff != 0]
    if len(diff) < 2:
        return {"W_stat": float("nan"), "p_value": float("nan")}
    result = sp_stats.wilcoxon(diff)
    return {"W_stat": float(result.statistic), "p_value": float(result.pvalue)}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size (pooled std). Positive = a > b."""
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if len(xa) < 2 or len(xb) < 2:
        return float("nan")
    var_a = float(xa.var(ddof=1))
    var_b = float(xb.var(ddof=1))
    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std < 1e-12:
        return float("nan")
    return float((xa.mean() - xb.mean()) / pooled_std)


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down correction. Returns corrected p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        adj = p * (n - rank + 1)
        corrected[orig_idx] = max(adj, prev)
        prev = corrected[orig_idx]
    return [min(p, 1.0) for p in corrected]


def pairwise_tests(
    records: List[Dict[str, object]],
    baseline_variant: str,
    metrics: Sequence[str],
    group_key: str = "variant",
    confidence: float = 0.95,
) -> List[Dict[str, object]]:
    """
    Compare every variant against *baseline_variant* on each metric.

    Returns rows with: metric, variant, baseline, delta_mean, cohens_d,
    t_stat, p_value_raw, p_value_corrected, ci_lower, ci_upper, significant.
    """
    grouped: Dict[str, List[Dict]] = {}
    for rec in records:
        key = str(rec.get(group_key, ""))
        grouped.setdefault(key, []).append(rec)

    if baseline_variant not in grouped:
        return []

    all_rows: List[Dict[str, object]] = []

    for metric in metrics:
        baseline_vals = [
            float(r[metric])
            for r in grouped[baseline_variant]
            if r.get(metric) is not None and not math.isnan(float(r[metric]))
        ]
        if len(baseline_vals) < 2:
            continue

        for variant, variant_records in sorted(grouped.items()):
            if variant == baseline_variant:
                continue
            variant_vals = [
                float(r[metric])
                for r in variant_records
                if r.get(metric) is not None and not math.isnan(float(r[metric]))
            ]
            if len(variant_vals) < 2:
                continue

            t_result = welch_t_test(variant_vals, baseline_vals)
            d = cohens_d(variant_vals, baseline_vals)
            delta = float(np.mean(variant_vals)) - float(np.mean(baseline_vals))
            _, ci_lo, ci_hi = confidence_interval(variant_vals, confidence)

            all_rows.append({
                "metric": metric,
                "variant": variant,
                "baseline": baseline_variant,
                "n_variant": len(variant_vals),
                "n_baseline": len(baseline_vals),
                "variant_mean": float(np.mean(variant_vals)),
                "baseline_mean": float(np.mean(baseline_vals)),
                "delta_mean": delta,
                "cohens_d": d,
                "t_stat": t_result["t_stat"],
                "p_value_raw": t_result["p_value"],
                "p_value_corrected": None,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "significant_005": None,
            })

    # Holm-Bonferroni correction across all tests
    if all_rows:
        raw_ps = [r["p_value_raw"] for r in all_rows]
        corrected_ps = holm_bonferroni(raw_ps)
        for row, cp in zip(all_rows, corrected_ps):
            row["p_value_corrected"] = cp
            row["significant_005"] = cp < 0.05

    return all_rows


def compute_confidence_intervals(
    records: List[Dict[str, object]],
    metrics: Sequence[str],
    group_key: str = "variant",
    confidence: float = 0.95,
) -> List[Dict[str, object]]:
    """Compute CI for each (variant, metric) pair."""
    grouped: Dict[str, List[Dict]] = {}
    for rec in records:
        key = str(rec.get(group_key, ""))
        grouped.setdefault(key, []).append(rec)

    rows: List[Dict[str, object]] = []
    for variant, variant_records in sorted(grouped.items()):
        for metric in metrics:
            vals = [
                float(r[metric])
                for r in variant_records
                if r.get(metric) is not None and not math.isnan(float(r[metric]))
            ]
            if not vals:
                continue
            mean, ci_lo, ci_hi = confidence_interval(vals, confidence)
            rows.append({
                "variant": variant,
                "metric": metric,
                "n": len(vals),
                "mean": mean,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "confidence": confidence,
            })
    return rows
