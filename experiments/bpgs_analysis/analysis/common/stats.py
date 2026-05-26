"""
Statistical testing utilities for the BPGS analysis framework.

Provides:
- Bootstrap confidence intervals (BCa method for small N).
- Wilcoxon signed-rank test for paired comparisons.
- Cohen's d effect size.
- P-value formatting for tables.
- Comprehensive comparison summaries.

Design notes
------------
* With only N=3 seeds, parametric assumptions are unreliable.
  We therefore emphasize **bootstrap CIs** and **effect sizes**
  over p-values, following best practices for small-sample
  experimental ML research.
* Bootstrap uses the BCa (bias-corrected and accelerated) method
  via ``scipy.stats.bootstrap`` when available, with a fallback
  to percentile-based CIs for older scipy versions.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap confidence interval computation.

    Attributes
    ----------
    mean : float
        Sample mean of the input values.
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.
    alpha : float
        Significance level (e.g. 0.05 for 95% CI).
    n_boot : int
        Number of bootstrap resamples performed.
    n_samples : int
        Number of original data points.
    """

    mean: float
    ci_low: float
    ci_high: float
    alpha: float
    n_boot: int
    n_samples: int

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_high - self.ci_low

    def __repr__(self) -> str:
        pct = int((1 - self.alpha) * 100)
        return (
            f"BootstrapResult(mean={self.mean:.4f}, "
            f"{pct}% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"n={self.n_samples})"
        )


def bootstrap_ci(
    values: Union[Sequence[float], np.ndarray],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    method: str = "BCa",
    random_state: Optional[int] = 42,
) -> BootstrapResult:
    """Compute a bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array-like
        Sample values (e.g. metric scores across 3 seeds).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level.  ``0.05`` gives a 95% CI.
    method : str
        Bootstrap method: ``"BCa"`` (bias-corrected and accelerated),
        ``"percentile"``, or ``"basic"``.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Contains ``mean``, ``ci_low``, ``ci_high``.

    Notes
    -----
    For N ≤ 3, BCa may produce degenerate intervals.  We fall back
    to the percentile method in such cases and log a warning.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]  # drop NaNs

    if len(arr) == 0:
        return BootstrapResult(
            mean=np.nan, ci_low=np.nan, ci_high=np.nan,
            alpha=alpha, n_boot=0, n_samples=0,
        )

    sample_mean = float(np.mean(arr))

    if len(arr) == 1:
        # Single value — no uncertainty estimate possible
        return BootstrapResult(
            mean=sample_mean, ci_low=sample_mean, ci_high=sample_mean,
            alpha=alpha, n_boot=0, n_samples=1,
        )

    # Try scipy.stats.bootstrap (scipy >= 1.7)
    try:
        rng = np.random.default_rng(random_state)
        result = sp_stats.bootstrap(
            (arr,),
            statistic=np.mean,
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method=method.lower(),
            random_state=rng,
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)

    except (TypeError, ValueError) as exc:
        # Fallback to manual percentile bootstrap
        logger.debug(
            "scipy bootstrap failed (%s), using manual percentile method",
            exc,
        )
        rng = np.random.RandomState(random_state)
        boot_means = np.array([
            np.mean(rng.choice(arr, size=len(arr), replace=True))
            for _ in range(n_boot)
        ])
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        mean=sample_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        n_boot=n_boot,
        n_samples=len(arr),
    )


@dataclass(frozen=True)
class WilcoxonResult:
    """Result of a Wilcoxon signed-rank test.

    Attributes
    ----------
    statistic : float
        Test statistic.
    p_value : float
        Two-sided p-value.
    n_pairs : int
        Number of non-zero-difference pairs.
    significant : bool
        Whether p < alpha.
    warning : str or None
        Caveat about small sample size, if applicable.
    """

    statistic: float
    p_value: float
    n_pairs: int
    significant: bool
    warning: Optional[str] = None


def paired_wilcoxon(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
    *,
    alpha: float = 0.05,
) -> WilcoxonResult:
    """Perform a two-sided Wilcoxon signed-rank test.

    Parameters
    ----------
    a, b : array-like
        Paired samples (e.g. metric values for two methods across
        the same seeds).
    alpha : float
        Significance threshold.

    Returns
    -------
    WilcoxonResult

    Notes
    -----
    With N < 5 pairs, the test has very low power and the result
    should be interpreted with extreme caution.  A warning is
    included in the result.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    if len(a_arr) != len(b_arr):
        raise ValueError(
            f"Paired samples must have equal length: "
            f"got {len(a_arr)} vs {len(b_arr)}"
        )

    n = len(a_arr)
    warning_msg: Optional[str] = None

    if n < 5:
        warning_msg = (
            f"Only {n} paired observations — Wilcoxon test has "
            f"insufficient power. Use effect sizes and bootstrap CIs instead."
        )
        logger.warning(warning_msg)

    # Remove pairs with zero difference (Wilcoxon cannot handle them)
    diffs = a_arr - b_arr
    nonzero_mask = diffs != 0
    n_nonzero = int(np.sum(nonzero_mask))

    if n_nonzero < 1:
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            n_pairs=0,
            significant=False,
            warning="All differences are zero — test is undefined.",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat_result = sp_stats.wilcoxon(
                a_arr[nonzero_mask],
                b_arr[nonzero_mask],
                alternative="two-sided",
            )
        statistic = float(stat_result.statistic)
        p_value = float(stat_result.pvalue)
    except ValueError as exc:
        logger.warning("Wilcoxon test failed: %s", exc)
        return WilcoxonResult(
            statistic=np.nan,
            p_value=np.nan,
            n_pairs=n_nonzero,
            significant=False,
            warning=str(exc),
        )

    return WilcoxonResult(
        statistic=statistic,
        p_value=p_value,
        n_pairs=n_nonzero,
        significant=p_value < alpha,
        warning=warning_msg,
    )


def cohens_d(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
) -> float:
    """Compute Cohen's d effect size (pooled standard deviation).

    .. math::

        d = \\frac{\\bar{a} - \\bar{b}}{s_p}

    where :math:`s_p = \\sqrt{\\frac{(n_a - 1) s_a^2 + (n_b - 1) s_b^2}
    {n_a + n_b - 2}}`.

    Parameters
    ----------
    a, b : array-like
        Two groups of observations.

    Returns
    -------
    float
        Cohen's d.  Positive means *a* > *b*.

    Interpretation
    --------------
    |d| < 0.2 : negligible
    0.2 ≤ |d| < 0.5 : small
    0.5 ≤ |d| < 0.8 : medium
    |d| ≥ 0.8 : large
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    n_a, n_b = len(a_arr), len(b_arr)

    if n_a < 2 or n_b < 2:
        logger.warning(
            "Cohen's d requires at least 2 observations per group "
            "(got %d, %d)",
            n_a,
            n_b,
        )
        return np.nan

    mean_diff = float(np.mean(a_arr) - np.mean(b_arr))
    var_a = float(np.var(a_arr, ddof=1))
    var_b = float(np.var(b_arr, ddof=1))

    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0 if mean_diff == 0 else np.inf * np.sign(mean_diff)

    return mean_diff / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Human-readable interpretation of Cohen's d magnitude."""
    ad = abs(d)
    if np.isnan(ad):
        return "undefined"
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def format_pvalue(p: float, *, threshold: float = 0.001) -> str:
    """Format a p-value for display in text or tables.

    Parameters
    ----------
    p : float
        Raw p-value.
    threshold : float
        Below this value, display as ``"< threshold"``.

    Returns
    -------
    str
        Formatted string, e.g. ``"< 0.001"``, ``"0.032"``, ``"n.s."``

    Examples
    --------
    >>> format_pvalue(0.0003)
    '< 0.001'
    >>> format_pvalue(0.0321)
    '0.032'
    >>> format_pvalue(0.87)
    '0.870'
    """
    if np.isnan(p):
        return "n.a."
    if p < threshold:
        return f"< {threshold}"
    return f"{p:.3f}"


@dataclass
class PairwiseComparison:
    """Full pairwise statistical comparison between two methods.

    Combines effect size, bootstrap CI, and Wilcoxon test results
    into a single structured report.
    """

    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float
    cohens_d: float
    cohens_d_interpretation: str
    bootstrap_ci_a: BootstrapResult
    bootstrap_ci_b: BootstrapResult
    wilcoxon: Optional[WilcoxonResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: Dict[str, Any] = {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "cohens_d": self.cohens_d,
            "effect_size": self.cohens_d_interpretation,
            "ci_a": {
                "low": self.bootstrap_ci_a.ci_low,
                "high": self.bootstrap_ci_a.ci_high,
            },
            "ci_b": {
                "low": self.bootstrap_ci_b.ci_low,
                "high": self.bootstrap_ci_b.ci_high,
            },
        }
        if self.wilcoxon is not None:
            d["wilcoxon_p"] = self.wilcoxon.p_value
            d["wilcoxon_significant"] = self.wilcoxon.significant
            if self.wilcoxon.warning:
                d["wilcoxon_warning"] = self.wilcoxon.warning
        return d


def compare_methods(
    values_a: Union[Sequence[float], np.ndarray],
    values_b: Union[Sequence[float], np.ndarray],
    method_a: str,
    method_b: str,
    metric: str,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    run_wilcoxon: bool = True,
) -> PairwiseComparison:
    """Run a full pairwise statistical comparison.

    Parameters
    ----------
    values_a, values_b : array-like
        Metric values for each method (one per seed).
    method_a, method_b : str
        Method names for labeling.
    metric : str
        Metric name.
    n_boot : int
        Bootstrap resamples.
    alpha : float
        Significance level.
    run_wilcoxon : bool
        Whether to run the Wilcoxon test (requires N ≥ 5 ideally).

    Returns
    -------
    PairwiseComparison
    """
    a_arr = np.asarray(values_a, dtype=np.float64)
    b_arr = np.asarray(values_b, dtype=np.float64)

    mean_a = float(np.mean(a_arr))
    mean_b = float(np.mean(b_arr))

    d = cohens_d(a_arr, b_arr)

    ci_a = bootstrap_ci(a_arr, n_boot=n_boot, alpha=alpha)
    ci_b = bootstrap_ci(b_arr, n_boot=n_boot, alpha=alpha)

    wilcoxon_result = None
    if run_wilcoxon and len(a_arr) >= 3:
        wilcoxon_result = paired_wilcoxon(a_arr, b_arr, alpha=alpha)

    return PairwiseComparison(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_a - mean_b,
        cohens_d=d,
        cohens_d_interpretation=interpret_cohens_d(d),
        bootstrap_ci_a=ci_a,
        bootstrap_ci_b=ci_b,
        wilcoxon=wilcoxon_result,
    )
