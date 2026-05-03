from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STUDY_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "studies" / "bpgs_objective"
SUBSET_ROOT = STUDY_ROOT / "assets" / "nyuv2_subsets" / "v1"


@dataclass(frozen=True)
class TrainingVariant:
    label: str
    dataset: str
    method: str
    epochs: int
    seeds: Sequence[int]
    extra_overrides: Sequence[str] = field(default_factory=list)
    subset_budget: str | None = None
    subset_seed: int | None = None
    use_subset_file: bool = False
    early_stop: bool = True
    early_stop_patience: int | None = None
    early_stop_min_delta: float | None = None


@dataclass(frozen=True)
class EmpiricalVariant:
    label: str
    experiment: str
    family: str
    methods: Sequence[str]
    seeds: Sequence[int]
    overrides: Sequence[str]


@dataclass(frozen=True)
class StudySpec:
    name: str
    kind: str  # training | empirical
    description: str
    variants: Sequence[TrainingVariant | EmpiricalVariant]
    requires_nyuv2_subsets: bool = False


def nyuv2_subset_id(budget: str, subset_seed: int) -> str:
    return f"nyuv2_train_p{budget}_s{subset_seed}_v1"


def nyuv2_subset_path(budget: str, subset_seed: int) -> Path:
    return SUBSET_ROOT / f"{nyuv2_subset_id(budget, subset_seed)}.json"


STUDIES: List[StudySpec] = [
    StudySpec(
        name="01_nyuv2_ablation",
        kind="training",
        description="NYUv2 subset ablation for BPGS mechanism choices.",
        requires_nyuv2_subsets=True,
        variants=[
            TrainingVariant(
                label="bpgs",
                dataset="nyuv2",
                method="bpgs",
                epochs=60,
                seeds=[42, 43, 44],
                subset_budget="50",
                subset_seed=11,
                use_subset_file=True,
                early_stop_patience=15,
                early_stop_min_delta=0.0001,
            ),
            TrainingVariant(
                label="bpgs_no_auto",
                dataset="nyuv2",
                method="bpgs",
                epochs=60,
                seeds=[42, 43, 44],
                extra_overrides=["method.auto_calibrate=false"],
                subset_budget="50",
                subset_seed=11,
                use_subset_file=True,
                early_stop_patience=15,
                early_stop_min_delta=0.0001,
            ),
            TrainingVariant(
                label="bpgs_fixed_temp",
                dataset="nyuv2",
                method="bpgs",
                epochs=60,
                seeds=[42, 43, 44],
                extra_overrides=["method.temperature=6.0"],
                subset_budget="50",
                subset_seed=11,
                use_subset_file=True,
                early_stop_patience=15,
                early_stop_min_delta=0.0001,
            ),
            TrainingVariant(
                label="bpgs_adaptive_temp",
                dataset="nyuv2",
                method="bpgs",
                epochs=60,
                seeds=[42, 43, 44],
                extra_overrides=["method.temperature=null"],
                subset_budget="50",
                subset_seed=11,
                use_subset_file=True,
                early_stop_patience=15,
                early_stop_min_delta=0.0001,
            ),
            TrainingVariant(
                label="kendall_unbounded",
                dataset="nyuv2",
                method="kendall",
                epochs=60,
                seeds=[42, 43, 44],
                subset_budget="50",
                subset_seed=11,
                use_subset_file=True,
                early_stop_patience=15,
                early_stop_min_delta=0.0001,
            ),
        ],
    ),
    StudySpec(
        name="02_synthetic_scale_stress",
        kind="empirical",
        description="Controlled synthetic scale-mismatch stress benchmark.",
        variants=[
            EmpiricalVariant(
                label="x1",
                experiment="exp_03_imbalance_robustness",
                family="imbalance",
                methods=["bpgs", "uwso", "kendall"],
                seeds=[42, 43, 44],
                overrides=["experiment.generator.imbalance_ratio=1.0"],
            ),
            EmpiricalVariant(
                label="x10",
                experiment="exp_03_imbalance_robustness",
                family="imbalance",
                methods=["bpgs", "uwso", "kendall"],
                seeds=[42, 43, 44],
                overrides=["experiment.generator.imbalance_ratio=10.0"],
            ),
            EmpiricalVariant(
                label="x100",
                experiment="exp_03_imbalance_robustness",
                family="imbalance",
                methods=["bpgs", "uwso", "kendall"],
                seeds=[42, 43, 44],
                overrides=["experiment.generator.imbalance_ratio=100.0"],
            ),
            EmpiricalVariant(
                label="x1000",
                experiment="exp_03_imbalance_robustness",
                family="imbalance",
                methods=["bpgs", "uwso", "kendall"],
                seeds=[42, 43, 44],
                overrides=["experiment.generator.imbalance_ratio=1000.0"],
            ),
        ],
    ),
    StudySpec(
        name="03_yeast_regime_check",
        kind="training",
        description="Cross-regime Yeast scouting comparison.",
        variants=[
            TrainingVariant(label="static", dataset="yeast", method="static", epochs=120, seeds=[42]),
            TrainingVariant(label="uwso", dataset="yeast", method="uwso", epochs=120, seeds=[42]),
            TrainingVariant(label="bpgs", dataset="yeast", method="bpgs", epochs=120, seeds=[42]),
        ],
    ),
    StudySpec(
        name="04_qm9_regime_check",
        kind="training",
        description="Cross-regime QM9 scouting comparison.",
        variants=[
            TrainingVariant(label="pcgrad", dataset="qm9", method="pcgrad", epochs=120, seeds=[42]),
            TrainingVariant(label="uwso", dataset="qm9", method="uwso", epochs=120, seeds=[42]),
            TrainingVariant(label="bpgs", dataset="qm9", method="bpgs", epochs=120, seeds=[42]),
        ],
    ),
    StudySpec(
        name="05_nyuv2_full_final",
        kind="training",
        description="Final full NYUv2 seeded comparison for the paper table.",
        variants=[
            TrainingVariant(label="static", dataset="nyuv2", method="static", epochs=120, seeds=[42, 43, 44], early_stop_patience=15, early_stop_min_delta=0.0001),
            TrainingVariant(label="kendall", dataset="nyuv2", method="kendall", epochs=120, seeds=[42, 43, 44], early_stop_patience=15, early_stop_min_delta=0.0001),
            TrainingVariant(label="uwso", dataset="nyuv2", method="uwso", epochs=120, seeds=[42, 43, 44], early_stop_patience=15, early_stop_min_delta=0.0001),
            TrainingVariant(label="bpgs", dataset="nyuv2", method="bpgs", epochs=120, seeds=[42, 43, 44], early_stop_patience=15, early_stop_min_delta=0.0001),
        ],
    ),
]


def get_study(name: str) -> StudySpec:
    for study in STUDIES:
        if study.name == name:
            return study
    raise KeyError(f"Unknown study: {name}")
