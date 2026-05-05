"""Comparative analysis across methods and seeds."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def summarize_comparison_frame(frame: pd.DataFrame, metric: str = "macro_score") -> pd.DataFrame:
    grouped = frame.groupby("method")[metric]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["stderr"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    return summary.sort_values("mean", ascending=False).reset_index(drop=True)


def paired_method_deltas(frame: pd.DataFrame, left: str, right: str, metric: str = "macro_score") -> Dict[str, float]:
    pivot = frame.pivot(index="seed", columns="method", values=metric)
    deltas = pivot[left] - pivot[right]
    return {
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=0)),
        "wins": float((deltas > 0).sum()),
    }

