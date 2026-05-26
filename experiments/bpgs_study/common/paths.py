from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "bpgs_study"
LEGACY_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "studies" / "bpgs_objective"
SUBSET_ROOT = STUDY_ROOT / "assets" / "nyuv2_subsets" / "v1"


def nyuv2_subset_id(budget: str, subset_seed: int) -> str:
    return f"nyuv2_train_p{budget}_s{subset_seed}_v1"


def nyuv2_subset_path(budget: str, subset_seed: int) -> Path:
    return SUBSET_ROOT / f"{nyuv2_subset_id(budget, subset_seed)}.json"


def resolve_study_output_root(study_name: str) -> Path:
    preferred = OUTPUT_ROOT / study_name
    if preferred.exists():
        return preferred
    legacy = LEGACY_OUTPUT_ROOT / study_name
    if legacy.exists():
        return legacy
    return preferred
