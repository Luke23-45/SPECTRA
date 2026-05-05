"""Task-level performance metrics for synthetic empirical benchmarks."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from spectra.empirical.generators.base import TaskSpec


def compute_task_performance(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    task_specs: Sequence[TaskSpec],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for spec in task_specs:
        y_true = np.asarray(targets[spec.name])
        y_pred = np.asarray(predictions[spec.name])
        if spec.task_type == "regression":
            mse = mean_squared_error(y_true, y_pred)
            r2 = float(r2_score(y_true, y_pred))
            metrics[spec.name] = {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": r2,
                "score": float(np.clip(r2, -1.0, 1.0)),
            }
        elif spec.task_type == "binary":
            probs = 1.0 / (1.0 + np.exp(-y_pred))
            labels = (probs >= 0.5).astype(np.int64)
            entry = {
                "accuracy": float(accuracy_score(y_true, labels)),
                "f1": float(f1_score(y_true, labels, zero_division=0)),
                "score": float(f1_score(y_true, labels, zero_division=0)),
            }
            if len(np.unique(y_true)) > 1:
                entry["auroc"] = float(roc_auc_score(y_true, probs))
            else:
                entry["auroc"] = float("nan")
            metrics[spec.name] = entry
        elif spec.task_type == "multiclass":
            labels = np.argmax(y_pred, axis=1)
            metrics[spec.name] = {
                "accuracy": float(accuracy_score(y_true, labels)),
                "macro_f1": float(f1_score(y_true, labels, average="macro", zero_division=0)),
                "score": float(f1_score(y_true, labels, average="macro", zero_division=0)),
            }
        else:
            pred_unit = y_pred / np.clip(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-8, None)
            true_unit = y_true / np.clip(np.linalg.norm(y_true, axis=1, keepdims=True), 1e-8, None)
            cosine = np.clip(np.sum(pred_unit * true_unit, axis=1), -1.0, 1.0)
            mean_angle_rad = float(np.mean(np.arccos(cosine)))
            mean_angle_deg = float(np.degrees(mean_angle_rad))
            metrics[spec.name] = {
                "mean_angle_deg": mean_angle_deg,
                "median_angle_deg": float(np.degrees(np.median(np.arccos(cosine)))),
                "score": float(np.clip(1.0 - (mean_angle_rad / np.pi), 0.0, 1.0)),
            }
    return metrics


def summarize_performance(task_metrics: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    scores = [float(metrics["score"]) for metrics in task_metrics.values()]
    return {
        "macro_score": float(np.mean(scores)),
        "worst_task_score": float(np.min(scores)),
        "best_task_score": float(np.max(scores)),
        "score_std": float(np.std(scores)),
    }

