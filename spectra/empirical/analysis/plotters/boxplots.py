"""Boxplot utilities for distributional comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_boxplot(frame: pd.DataFrame, output_path: str | Path, metric: str = "macro_score") -> str:
    plt.figure(figsize=(6, 4))
    frame.boxplot(column=metric, by="method")
    plt.title(metric)
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)

