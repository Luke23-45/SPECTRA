"""Shared primitives for BPGS study execution.

Common provides only data types, paths, config loading, and selection
helpers.  All execution planning and command building is owned by each
study group's own runner and commands modules.
"""

from experiments.bpgs_study.common.paths import OUTPUT_ROOT, PROJECT_ROOT, STUDY_ROOT
from experiments.bpgs_study.common.catalog import get_registered_studies
from experiments.bpgs_study.common.runtime import RunRequest, Stage
from experiments.bpgs_study.common.specs import EmpiricalVariant, OverrideSpec, StudySpec, TrainingVariant

__all__ = [
    "EmpiricalVariant",
    "OverrideSpec",
    "OUTPUT_ROOT",
    "PROJECT_ROOT",
    "RunRequest",
    "Stage",
    "STUDY_ROOT",
    "StudySpec",
    "TrainingVariant",
    "get_registered_studies",
]
