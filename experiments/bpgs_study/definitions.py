from __future__ import annotations

from experiments.bpgs_study.common.paths import OUTPUT_ROOT, STUDY_ROOT, SUBSET_ROOT, nyuv2_subset_id, nyuv2_subset_path
from experiments.bpgs_study.common.catalog import get_registered_studies, get_study_entry
from experiments.bpgs_study.common.specs import EmpiricalVariant, OverrideSpec, StudySpec, TrainingVariant

STUDIES = get_registered_studies()


def get_study(name: str) -> StudySpec:
    return get_study_entry(name).spec


__all__ = [
    "EmpiricalVariant",
    "OverrideSpec",
    "OUTPUT_ROOT",
    "STUDIES",
    "STUDY_ROOT",
    "SUBSET_ROOT",
    "StudySpec",
    "TrainingVariant",
    "get_study",
    "nyuv2_subset_id",
    "nyuv2_subset_path",
]
