"""Stability and gradient-dynamics metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from spectra.empirical.metrics.trajectory import compute_oscillation_metrics, compute_smoothness_metrics


def compute_variance_metrics(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "variance": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "variance": float(np.var(array)),
        "std": float(np.std(array)),
    }


def compute_stability_metrics(history: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    train_losses = [float(epoch["train_total_loss"]) for epoch in history]
    val_losses = [float(epoch["val_total_loss"]) for epoch in history]
    grad_norms = [float(epoch.get("gradient_summary", {}).get("mean_grad_norm", np.nan)) for epoch in history]
    cosines = [float(epoch.get("gradient_summary", {}).get("mean_pairwise_cosine", np.nan)) for epoch in history]

    metrics: Dict[str, float] = {}
    for prefix, values in (
        ("train_loss", train_losses),
        ("val_loss", val_losses),
        ("grad_norm", grad_norms),
        ("grad_cosine", cosines),
    ):
        for key, value in compute_variance_metrics(values).items():
            metrics[f"{prefix}_{key}"] = value

    metrics.update({f"train_{k}": v for k, v in compute_smoothness_metrics(train_losses).items()})
    metrics.update({f"val_{k}": v for k, v in compute_smoothness_metrics(val_losses).items()})
    metrics.update({f"train_{k}": v for k, v in compute_oscillation_metrics(train_losses).items()})
    metrics.update({f"val_{k}": v for k, v in compute_oscillation_metrics(val_losses).items()})
    return metrics

