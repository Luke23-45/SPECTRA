from __future__ import annotations

from studies.bpgs_study.common.config_loader import load_study_spec
from studies.bpgs_study.common.runtime import RunRequest, Stage, select_seeds, select_variants
from studies.bpgs_study.common.specs import StudySpec, TrainingVariant
from studies.bpgs_study.common.validation import validate_run_request
from studies.bpgs_study.optional.commands import build_training_stage
from studies.bpgs_study.optional.utils import CONFIG_DIR, CONFIG_FILES


class OptionalRunner:
    """Owns all execution planning for optional studies.

    The optional runner has full control over:
    - how training commands are built (via its own commands.py)
    - which overrides apply to which variants (method-scoping)
    - optional-specific validation
    """

    def load_studies(self) -> list[StudySpec]:
        return [load_study_spec(CONFIG_DIR / filename) for filename in CONFIG_FILES]

    def plan(self, study: StudySpec, request: RunRequest) -> list[Stage]:
        if study.kind != "training":
            raise TypeError(
                f"Optional study '{study.name}' must be a training study, "
                f"got kind='{study.kind}'."
            )

        validate_run_request(request)

        stages: list[Stage] = []
        for variant in select_variants(study, request.variant_labels):
            if not isinstance(variant, TrainingVariant):
                raise TypeError(
                    f"Optional variant '{variant.label}' must be a TrainingVariant."
                )
            seeds = select_seeds(variant.seeds, request.seeds)
            for seed in seeds:
                stages.append(
                    build_training_stage(study.name, variant, seed, request.overrides)
                )

        return stages
