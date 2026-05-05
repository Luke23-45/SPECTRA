from __future__ import annotations

from studies.bpgs_study.common.config_loader import load_study_spec
from studies.bpgs_study.common.runtime import RunRequest, Stage, select_seeds, select_variants
from studies.bpgs_study.common.specs import EmpiricalVariant, StudySpec
from studies.bpgs_study.common.validation import validate_run_request
from studies.bpgs_study.stress.commands import build_empirical_stage
from studies.bpgs_study.stress.utils import CONFIG_DIR, CONFIG_FILES


class StressRunner:
    """Owns all execution planning for stress studies.

    The stress runner has full control over:
    - how empirical experiment commands are built (via its own commands.py)
    - which overrides apply to which method (method-scoping)
    - stress-specific validation
    """

    def load_studies(self) -> list[StudySpec]:
        return [load_study_spec(CONFIG_DIR / filename) for filename in CONFIG_FILES]

    def plan(self, study: StudySpec, request: RunRequest) -> list[Stage]:
        if study.kind != "empirical":
            raise TypeError(
                f"Stress study '{study.name}' must be an empirical study, "
                f"got kind='{study.kind}'."
            )

        validate_run_request(request)

        stages: list[Stage] = []
        for variant in select_variants(study, request.variant_labels):
            if not isinstance(variant, EmpiricalVariant):
                raise TypeError(
                    f"Stress variant '{variant.label}' must be an EmpiricalVariant."
                )
            seeds = select_seeds(variant.seeds, request.seeds)
            for method in variant.methods:
                for seed in seeds:
                    stages.append(
                        build_empirical_stage(
                            study.name, variant, method, seed, request.overrides
                        )
                    )

        return stages
