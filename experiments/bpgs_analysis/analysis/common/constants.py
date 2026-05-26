"""
Canonical constants for the BPGS analysis framework.

This module is the **single source of truth** for:
- Method display names, ordering, colors, and markers
- Seed values
- Metric direction registry (higher-is-better vs lower-is-better)
- Study-specific path templates and dataset metadata

Every table, figure, and statistical test imports from here to ensure
consistency across the entire paper.

Design notes
------------
* Colors are chosen from a colorblind-safe palette verified with the
  Coblis Color Blindness Simulator (https://www.color-blindness.com/coblis/).
* Method ordering follows the paper convention: baselines first
  (alphabetical), then BPGS variants (canonical first, then alphabetical).
* Metric directions are used by ``latex.bold_best_in_column`` and
  ``metrics.compute_delta_m`` to correctly identify "best" values.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple


SEEDS: Tuple[int, ...] = (42, 43, 44)
"""Canonical seed values used across all experiments."""

N_SEEDS: int = len(SEEDS)


# Display names (paper-friendly, LaTeX-safe)
METHOD_DISPLAY: Dict[str, str] = OrderedDict([
    # Baselines (alphabetical)
    ("static",              "Static"),
    ("kendall",             "Kendall"),
    ("uwso",                "UWSO"),
    ("pcgrad",              "PCGrad"),
    ("gradnorm_proxy",      "GradNorm"),
    # BPGS variants
    ("bpgs",                "BPGS"),
    ("bpgs_canonical",      "BPGS (canonical)"),
    ("bpgs_batch_aware_fixed", "BPGS (BA-fixed)"),
    ("bpgs_stateless_auto", "BPGS (SL-auto)"),
    ("bpgs_stateless_fixed", "BPGS (SL-fixed)"),
])

# Canonical ordering for tables/legends: baselines first, then BPGS family.
METHOD_ORDER: Tuple[str, ...] = (
    "static",
    "kendall",
    "uwso",
    "pcgrad",
    "gradnorm_proxy",
    "bpgs",
    "bpgs_canonical",
    "bpgs_batch_aware_fixed",
    "bpgs_stateless_auto",
    "bpgs_stateless_fixed",
)

# Colorblind-safe palette (Tol + Wong, verified with Coblis simulator)
METHOD_COLORS: Dict[str, str] = {
    # Baselines — cool, muted tones
    "static":               "#88CCEE",   # sky blue
    "kendall":              "#332288",   # indigo
    "uwso":                 "#117733",   # forest green
    "pcgrad":               "#882255",   # wine
    "gradnorm_proxy":       "#CC6677",   # rose
    # BPGS family — warm, prominent
    "bpgs":                 "#EE3377",   # magenta-pink (hero)
    "bpgs_canonical":       "#EE3377",   # same hero
    "bpgs_batch_aware_fixed": "#EE7733", # tangerine
    "bpgs_stateless_auto":  "#CCBB44",   # olive gold
    "bpgs_stateless_fixed": "#44AA99",   # teal
}

# Markers for line/scatter plots
METHOD_MARKERS: Dict[str, str] = {
    "static":               "s",    # square
    "kendall":              "^",    # triangle up
    "uwso":                 "v",    # triangle down
    "pcgrad":               "D",    # diamond
    "gradnorm_proxy":       "P",    # plus (filled)
    "bpgs":                 "o",    # circle
    "bpgs_canonical":       "o",
    "bpgs_batch_aware_fixed": "X",  # X (filled)
    "bpgs_stateless_auto":  "p",    # pentagon
    "bpgs_stateless_fixed": "h",    # hexagon
}

# Line styles for distinguishing overlapping methods
METHOD_LINESTYLES: Dict[str, str] = {
    "static":               "-",
    "kendall":              "-",
    "uwso":                 "-",
    "pcgrad":               "-",
    "gradnorm_proxy":       "-",
    "bpgs":                 "-",
    "bpgs_canonical":       "-",
    "bpgs_batch_aware_fixed": "--",
    "bpgs_stateless_auto":  "-.",
    "bpgs_stateless_fixed": ":",
}


class MetricDir(str, Enum):
    """Whether higher or lower values indicate better performance."""
    MAX = "max"   # higher is better  (e.g. mIoU, R², F1)
    MIN = "min"   # lower  is better  (e.g. loss, RMSE, angle)


METRIC_DIRECTION: Dict[str, MetricDir] = {
    # NYUv2 validation metrics
    "val/miou":                     MetricDir.MAX,
    "val/segmentation_miou":        MetricDir.MAX,
    "val/segmentation_pixel_acc":   MetricDir.MAX,
    "val/depth_abs_rel":            MetricDir.MIN,
    "val/depth_rmse":               MetricDir.MIN,
    "val/depth_loss":               MetricDir.MIN,
    "val/normals_mean_angle":       MetricDir.MIN,
    "val/normals_within_11_25":     MetricDir.MAX,
    "val/normals_loss":             MetricDir.MIN,
    "val/segmentation_loss":        MetricDir.MIN,
    "val/total_loss":               MetricDir.MIN,

    # RF1 validation metrics
    "val/r2":                       MetricDir.MAX,
    "val/mae":                      MetricDir.MIN,
    "val/rmse":                     MetricDir.MIN,

    # Yeast classification metrics
    "val/macro_f1":                 MetricDir.MAX,
    "val/micro_f1":                 MetricDir.MAX,
    "val/hamming_acc":              MetricDir.MAX,
    "val/subset_acc":               MetricDir.MAX,

    # Stress test metrics
    "macro_score":                  MetricDir.MAX,
    "worst_task_score":             MetricDir.MAX,
    "best_task_score":              MetricDir.MAX,
    "score_std":                    MetricDir.MIN,   # lower variance → better

    # Training losses (always lower-is-better)
    "train/total_loss_epoch":       MetricDir.MIN,
    "train/total_loss_step":        MetricDir.MIN,
    "train/depth_loss":             MetricDir.MIN,
    "train/normals_loss":           MetricDir.MIN,
    "train/segmentation_loss":      MetricDir.MIN,
}


def get_metric_direction(metric: str) -> MetricDir:
    """Look up the direction for a metric, with intelligent fallback.

    For per-task RF1 metrics like ``val/flow_site_1_plus_48h_r2`` or
    Yeast per-label metrics, we infer direction from the suffix.

    Parameters
    ----------
    metric : str
        Full metric column name.

    Returns
    -------
    MetricDir
        Whether higher or lower is better.

    Raises
    ------
    KeyError
        If the metric is not in the registry and cannot be inferred.
    """
    if metric in METRIC_DIRECTION:
        return METRIC_DIRECTION[metric]

    # Heuristic: infer from suffix
    lower_metric = metric.lower()
    if any(s in lower_metric for s in ("loss", "rmse", "mae", "abs_rel", "angle", "error")):
        return MetricDir.MIN
    if any(s in lower_metric for s in ("r2", "acc", "f1", "auc", "iou", "score", "within")):
        return MetricDir.MAX

    raise KeyError(
        f"Unknown metric direction for '{metric}'. "
        f"Add it to METRIC_DIRECTION in constants.py or ensure it has "
        f"a recognizable suffix (loss/rmse/mae/r2/acc/f1/auc/iou/score)."
    )


# NYUv2 tasks (used for ΔM computation)

@dataclass(frozen=True)
class TaskSpec:
    """Specification for a single task in a multi-task benchmark."""
    name: str
    metric: str
    direction: MetricDir
    display_name: str


NYUV2_TASKS: Tuple[TaskSpec, ...] = (
    TaskSpec("segmentation", "val/segmentation_miou", MetricDir.MAX, r"mIoU $\uparrow$"),
    TaskSpec("depth",        "val/depth_abs_rel",     MetricDir.MIN, r"Abs Rel $\downarrow$"),
    TaskSpec("normals",      "val/normals_mean_angle", MetricDir.MIN, r"Angle $\downarrow$"),
)

# NYUv2 table columns (full set for main results table)
NYUV2_TABLE_METRICS: Tuple[Tuple[str, str, MetricDir], ...] = (
    ("val/segmentation_miou",      r"mIoU $\uparrow$",            MetricDir.MAX),
    ("val/depth_abs_rel",          r"Abs Rel $\downarrow$",       MetricDir.MIN),
    ("val/depth_rmse",             r"RMSE $\downarrow$",          MetricDir.MIN),
    ("val/normals_mean_angle",     r"Angle $\downarrow$",         MetricDir.MIN),
    ("val/normals_within_11_25",   r"Within 11.25$^\circ$ $\uparrow$",   MetricDir.MAX),
    ("val/total_loss",             r"Total Loss $\downarrow$",    MetricDir.MIN),
)

# Ablation study methods
ABLATION_METHODS: Tuple[str, ...] = (
    "bpgs_canonical",
    "bpgs_batch_aware_fixed",
    "bpgs_stateless_auto",
    "bpgs_stateless_fixed",
    "kendall",
)

# NYUv2 benchmark methods
NYUV2_METHODS: Tuple[str, ...] = (
    "bpgs",
    "kendall",
    "static",
    "uwso",
)

# Full-data methods
FULL_DATA_METHODS: Tuple[str, ...] = (
    "bpgs",
    "gradnorm_proxy",
    "kendall",
    "pcgrad",
    "static",
    "uwso",
)

# Stress test methods (per sub-experiment)
STRESS_SCALE_METHODS: Tuple[str, ...] = ("bpgs", "kendall", "uwso")
STRESS_HETERO_METHODS: Tuple[str, ...] = ("bpgs", "kendall", "pcgrad", "uwso")

# Stress test scales and regimes
STRESS_SCALES: Tuple[int, ...] = (1, 10, 100, 1000)
STRESS_REGIMES: Tuple[str, ...] = ("clean", "noisy", "conflict")

# NYUv2 nesting patterns (for path discovery)
NYUV2_DOUBLE_NESTED: FrozenSet[str] = frozenset({"bpgs", "static"})
"""Methods that use double-nested paths: ``method/method/seed_N/``."""

NYUV2_SINGLE_NESTED: FrozenSet[str] = frozenset({"kendall", "uwso"})
"""Methods that use single-nested paths: ``method/seed_N/``."""

# Ablation run-ID prefix mapping
ABLATION_RUN_PREFIX: Dict[str, str] = {
    "bpgs_canonical":       "bpgs",
    "bpgs_batch_aware_fixed": "bpgs",
    "bpgs_stateless_auto":  "bpgs",
    "bpgs_stateless_fixed": "bpgs",
    "kendall":              "kendall",
}
"""Maps ablation method name → run_id prefix used in directory names.
ALL BPGS variants use the prefix ``bpgs``, not the variant name."""

@dataclass(frozen=True)
class EpochSpec:
    """Training configuration for a study/dataset."""
    max_epochs: int
    patience: int
    typical_stopped: str   # human-readable description


EPOCH_SPECS: Dict[str, EpochSpec] = {
    "ablation":   EpochSpec(60,  15, "60 (no early stop)"),
    "nyuv2":      EpochSpec(120, 15, "120 (no early stop)"),
    "yeast":      EpochSpec(120, 15, "~23 (early stopped)"),
    "rf1":        EpochSpec(120, 15, "varies"),
}


# Directory layout constants — loaded from configs/base.yaml via OmegaConf.

def _load_path_constants():
    """Load directory layout constants from base.yaml config."""
    try:
        from analysis.common.config import get_config
        cfg = get_config()
        return {
            "ABLATION_BASE_DIR":       Path(cfg.paths.ablation),
            "NYUV2_BASE_DIR":          Path(cfg.paths.nyuv2),
            "YEAST_BASE_DIR":          Path(cfg.paths.yeast),
            "RF1_BASE_DIR":            Path(cfg.paths.rf1),
            "STRESS_SCALE_BASE_DIR":   Path(cfg.paths.stress_scale),
            "STRESS_RESCALE_BASE_DIR": Path(cfg.paths.stress_rescale),
            "STRESS_HETERO_BASE_DIR":  Path(cfg.paths.stress_hetero),
        }
    except Exception:
        # Fallback to hardcoded values if config loading fails
        # (e.g. during initial import before omegaconf is installed)
        return {
            "ABLATION_BASE_DIR":       Path("abalation/final_outputs"),
            "NYUV2_BASE_DIR":          Path("nyuv2/final_output_versions"),
            "YEAST_BASE_DIR":          Path("full_data/final_results/03_yeast_regime_check"),
            "RF1_BASE_DIR":            Path("full_data/final_results/08_rf1_regime_check"),
            "STRESS_SCALE_BASE_DIR":   Path("stress/final_outputs/02_synthetic_scale_stress"),
            "STRESS_RESCALE_BASE_DIR": Path("stress/final_outputs/06_pure_loss_rescaling"),
            "STRESS_HETERO_BASE_DIR":  Path("stress/final_outputs/07_heterogeneous_mixed_stress"),
        }


# Lazy-load path constants on first access
class _PathConstants:
    """Lazy-loading container for path constants from config.

    Defers config loading until a path constant is actually accessed,
    avoiding circular import issues during module initialization.
    """

    _loaded: bool = False
    _paths: Dict[str, Path] = {}

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._loaded:
            cls._paths = _load_path_constants()
            cls._loaded = True

    @classmethod
    def get(cls, name: str) -> Path:
        cls._ensure_loaded()
        return cls._paths[name]


# Module-level aliases for backward compatibility.
# These are accessed as regular module attributes via __getattr__.
_PATH_CONSTANT_NAMES = {
    "ABLATION_BASE_DIR",
    "NYUV2_BASE_DIR",
    "YEAST_BASE_DIR",
    "RF1_BASE_DIR",
    "STRESS_SCALE_BASE_DIR",
    "STRESS_RESCALE_BASE_DIR",
    "STRESS_HETERO_BASE_DIR",
}


def __getattr__(name: str):
    """Module-level __getattr__ for lazy path constant loading."""
    if name in _PATH_CONSTANT_NAMES:
        return _PathConstants.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Directories to ignore during path discovery (metadata-only)
STRESS_IGNORE_DIRS: FrozenSet[str] = frozenset({
    "exp_03_imbalance_robustness",
    "exp_07_pure_loss_rescaling",
    "exp_08_heterogeneous_regime",
})

def display_name(method: str) -> str:
    """Return the paper-friendly display name for a method.

    Falls back to title-casing the raw name if not in the registry.
    """
    return METHOD_DISPLAY.get(method, method.replace("_", " ").title())


def sort_methods(methods: Sequence[str]) -> List[str]:
    """Sort methods according to the canonical ``METHOD_ORDER``.

    Methods not in ``METHOD_ORDER`` are appended at the end in their
    original order.
    """
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    known = sorted(
        [m for m in methods if m in order_map],
        key=lambda m: order_map[m],
    )
    unknown = [m for m in methods if m not in order_map]
    return known + unknown
