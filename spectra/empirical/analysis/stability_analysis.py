"""Stability analysis helpers over runner outputs."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import pandas as pd

from spectra.empirical.metrics import compute_stability_metrics


def analyze_stability_histories(results: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"method": result["method"], "seed": result["seed"]}
        row.update(compute_stability_metrics(result["history"]))
        rows.append(row)
    return pd.DataFrame(rows)

