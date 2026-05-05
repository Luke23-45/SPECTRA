"""Robustness metrics under scale and class imbalance stress."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def compute_degradation(reference_score: float, stressed_score: float) -> float:
    if np.isnan(reference_score) or abs(reference_score) < 1e-12:
        return float("nan")
    return float((reference_score - stressed_score) / abs(reference_score))


def compute_task_dominance(task_scores: Mapping[str, float]) -> float:
    scores = np.asarray(list(task_scores.values()), dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    shifted = scores - scores.min() + 1e-8
    shifted = shifted / shifted.sum()
    return float(shifted.max())


def compute_robustness_metrics(reference_summary: Mapping[str, float], stressed_summary: Mapping[str, float]) -> Dict[str, float]:
    return {
        "macro_score_drop": compute_degradation(reference_summary["macro_score"], stressed_summary["macro_score"]),
        "worst_task_drop": compute_degradation(reference_summary["worst_task_score"], stressed_summary["worst_task_score"]),
        "score_dispersion_delta": float(stressed_summary["score_std"] - reference_summary["score_std"]),
    }

