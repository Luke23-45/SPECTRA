"""
ΔM (multi-task performance) computation for NYUv2.

Standalone module that wraps the generic ``metrics.compute_delta_m``
with NYUv2-specific task definitions for direct use by scripts.

The standard MTL ΔM formula:

    ΔM = (1/T) Σ_t  (-1)^{1[lower is better]}  ×  (M_t − M_t^base) / M_t^base

Using ``static`` as baseline.
Tasks: mIoU (↑), Abs Rel (↓), Normal Angle (↓).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

from analysis.common.constants import NYUV2_METHODS, NYUV2_TASKS, SEEDS
from analysis.common.metrics import compute_delta_m_for_nyuv2
from analysis.studies.nyuv2.extract import extract_nyuv2_data

logger = logging.getLogger(__name__)


def compute_all_delta_m(
    data_root: Optional[Path] = None,
    methods: Sequence[str] = NYUV2_METHODS,
    seeds: Sequence[int] = SEEDS,
    baseline_method: str = "static",
) -> Dict[str, float]:
    """Compute ΔM for all NYUv2 methods.

    Parameters
    ----------
    data_root : Path, optional
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str
        Methods to evaluate.
    seeds : sequence of int
        Seeds for aggregation.
    baseline_method : str
        Baseline method name.

    Returns
    -------
    dict[str, float]
        ``method → ΔM`` value.
    """
    data = extract_nyuv2_data(data_root, methods, seeds, baseline_method)
    return data["delta_m"]
