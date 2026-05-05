from __future__ import annotations

from studies.bpgs_study.ablation.commands import build_subset_stage, build_training_stage
from studies.bpgs_study.common.paths import nyuv2_subset_path
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

    @staticmethod
    def _required_subset_paths(
        variants: tuple[TrainingVariant, ...],
    ) -> tuple:
        paths = []
        for variant in variants:
            if variant.use_subset_file and variant.subset_budget and variant.subset_seed is not None:
                paths.append(nyuv2_subset_path(variant.subset_budget, variant.subset_seed))
        return tuple(dict.fromkeys(paths))

    def plan(self, study: StudySpec, request: RunRequest) -> list[Stage]:
        if study.kind != "training":
            raise TypeError(
                f"Ablation study '{study.name}' must be a training study, "
                f"got kind='{study.kind}'."
            )

        validate_run_request(request)

        selected_variants = select_variants(study, request.variant_labels)
        training_variants: tuple[TrainingVariant, ...] = tuple(selected_variants)  # validated below
        required_paths = self._required_subset_paths(training_variants)
        stages: list[Stage] = []
        if study.requires_nyuv2_subsets and not request.skip_prep:
            if any(not path.exists() for path in required_paths):
                stages.append(build_subset_stage(study.name))
        elif study.requires_nyuv2_subsets and request.skip_prep:
            missing_paths = [path for path in required_paths if not path.exists()]
            if missing_paths:
                missing_text = ", ".join(str(path) for path in missing_paths)
                raise ValueError(
                    "NYUv2 subset preparation was skipped, but required subset files are missing: "
                    f"{missing_text}"
                )

        for variant in training_variants:
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
