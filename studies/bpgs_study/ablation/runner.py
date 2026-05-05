from __future__ import annotations

from studies.bpgs_study.ablation.commands import build_subset_stage, build_training_stage
from studies.bpgs_study.ablation.utils import CONFIG_DIR, CONFIG_FILES
from studies.bpgs_study.common.config_loader import load_study_spec
from studies.bpgs_study.common.runtime import RunRequest, Stage, select_seeds, select_variants
from studies.bpgs_study.common.specs import StudySpec, TrainingVariant
from studies.bpgs_study.common.validation import validate_run_request


class AblationRunner:
    """Owns all execution planning for ablation studies.

    The ablation runner has full control over:
    - how training commands are built (via its own commands.py)
    - which overrides apply to which variants (method-scoping)
    - what pre-stages to include (e.g. NYUv2 subset preparation)
    - ablation-specific validation
    """

    def load_studies(self) -> list[StudySpec]:
        return [load_study_spec(CONFIG_DIR / filename) for filename in CONFIG_FILES]

    def plan(self, study: StudySpec, request: RunRequest) -> list[Stage]:
        if study.kind != "training":
            raise TypeError(
                f"Ablation study '{study.name}' must be a training study, "
                f"got kind='{study.kind}'."
            )

        validate_run_request(request)

        stages: list[Stage] = []
        if study.requires_nyuv2_subsets:
            stages.append(build_subset_stage(study.name))

        for variant in select_variants(study, request.variant_labels):
            if not isinstance(variant, TrainingVariant):
                raise TypeError(
                    f"Ablation variant '{variant.label}' must be a TrainingVariant."
                )
            seeds = select_seeds(variant.seeds, request.seeds)
            for seed in seeds:
                stages.append(
                    build_training_stage(study.name, variant, seed, request.overrides)
                )

        return stages
