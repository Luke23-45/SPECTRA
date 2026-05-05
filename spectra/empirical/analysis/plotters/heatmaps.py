"""Heatmap utilities for method-by-metric summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_heatmap(frame: pd.DataFrame, output_path: str | Path, value_column: str = "macro_score") -> str:
    pivot = frame.pivot_table(index="method", columns="seed", values=value_column, aggfunc="mean")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label=value_column)
    plt.xticks(range(len(pivot.columns)), [str(column) for column in pivot.columns])
    plt.yticks(range(len(pivot.index)), list(pivot.index))
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)

