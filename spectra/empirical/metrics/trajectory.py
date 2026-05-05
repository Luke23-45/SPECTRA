"""Time-series metrics for training trajectories."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.signal import find_peaks


def _series(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def compute_smoothness_metrics(values: Sequence[float]) -> Dict[str, float]:
    series = _series(values)
    if series.size < 2:
        return {"smoothness_l2": 0.0, "total_variation": 0.0}
    diffs = np.diff(series)
    return {
        "smoothness_l2": float(np.mean(diffs**2)),
        "total_variation": float(np.abs(diffs).sum()),
    }


def compute_oscillation_metrics(values: Sequence[float]) -> Dict[str, float]:
    series = _series(values)
    if series.size < 3:
        return {"zero_crossing_rate": 0.0, "peak_count": 0.0}
    centered = series - np.mean(series)
    zero_crossings = np.sum(np.signbit(centered[1:]) != np.signbit(centered[:-1]))
    peaks, _ = find_peaks(series)
    troughs, _ = find_peaks(-series)
    return {
        "zero_crossing_rate": float(zero_crossings / max(1, series.size - 1)),
        "peak_count": float(len(peaks) + len(troughs)),
    }

