"""Boundedness metrics for BPGS-style latent precision control."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def compute_entropy(weights: Sequence[float]) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = np.clip(values, 1e-12, None)
    values = values / values.sum()
    return float(-(values * np.log(values)).sum())


def compute_boundedness_metrics(history: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    weight_vectors = []
    latent_vectors = []
    theta_vectors = []
    for epoch in history:
        weights = epoch.get("task_weights", {})
        if weights:
            ordered_weights = [float(value) for _, value in sorted(weights.items())]
            weight_vectors.append(ordered_weights)
        latent = epoch.get("latent_state", {})
        if latent:
            s_values = [float(value) for key, value in sorted(latent.items()) if str(key).startswith("s_")]
            theta_values = [float(value) for key, value in sorted(latent.items()) if str(key).startswith("theta_")]
            if s_values:
                latent_vectors.append(s_values)
            if theta_values:
                theta_vectors.append(theta_values)

    metrics: Dict[str, float] = {}
    if weight_vectors:
        weights_np = np.asarray(weight_vectors, dtype=np.float64)
        metrics["weight_min"] = float(weights_np.min())
        metrics["weight_max"] = float(weights_np.max())
        metrics["weight_entropy_mean"] = float(np.mean([compute_entropy(row) for row in weights_np]))
        metrics["weight_entropy_min"] = float(np.min([compute_entropy(row) for row in weights_np]))
    if latent_vectors:
        latent_np = np.asarray(latent_vectors, dtype=np.float64)
        metrics["latent_min"] = float(latent_np.min())
        metrics["latent_max"] = float(latent_np.max())
        metrics["latent_span"] = float(latent_np.max() - latent_np.min())
    if theta_vectors:
        theta_np = np.asarray(theta_vectors, dtype=np.float64)
        metrics["theta_min"] = float(theta_np.min())
        metrics["theta_max"] = float(theta_np.max())
        metrics["theta_span"] = float(theta_np.max() - theta_np.min())
    return metrics

