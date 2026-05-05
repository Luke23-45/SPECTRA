from __future__ import annotations

import sys
from pathlib import Path

from studies.bpgs_study.common.paths import OUTPUT_ROOT, PROJECT_ROOT
from studies.bpgs_study.common.runtime import Stage
from studies.bpgs_study.common.specs import OverrideSpec, TrainingVariant


def _resolve_output_dir(study_name: str, variant: TrainingVariant, seed: int) -> Path:
    return OUTPUT_ROOT / study_name / variant.label / f"seed_{seed}"


def build_training_stage(
    study_name: str,
    variant: TrainingVariant,
    seed: int,
    overrides: tuple[OverrideSpec, ...],
) -> Stage:
    """Build a Hydra training command for a single variant-seed pair.

    The real_data runner owns this logic entirely.  Overrides are
    method-scoped: only overrides whose ``scope_method`` matches
    ``variant.method`` (or have no scope) are applied.
    """
    from studies.bpgs_study.common.runtime import filter_overrides_for_method

    applicable = filter_overrides_for_method(overrides, variant.method)

    cmd = [sys.executable, "-m", "scripts.run_training"]
    cmd.append(f"dataset={variant.dataset}")
    cmd.append(f"method={variant.method}")
    cmd.append(f"train.epochs={variant.epochs}")
    cmd.append(f"seed={seed}")
    cmd.append("require_cuda=true")

    if variant.early_stop:
        cmd.append("train.early_stop=true")
        if variant.early_stop_patience is not None:
            cmd.append(f"train.early_stop_patience={variant.early_stop_patience}")
        if variant.early_stop_min_delta is not None:
            cmd.append(f"train.early_stop_min_delta={variant.early_stop_min_delta}")
    else:
        cmd.append("train.early_stop=false")

    if variant.use_subset_file and variant.subset_budget and variant.subset_seed is not None:
        from studies.bpgs_study.common.paths import nyuv2_subset_path
        subset_path = nyuv2_subset_path(variant.subset_budget, variant.subset_seed)
        cmd.append(f"train_subset_file={subset_path}")

    for extra in variant.extra_overrides:
        cmd.append(str(extra))

    for override in applicable:
        cmd.append(override.to_hydra_arg())

    output_dir = _resolve_output_dir(study_name, variant, seed)
    cmd.append(f"output_dir={output_dir}")
    cmd.append(f"hydra.run.dir={output_dir}")

    return Stage(
        label=f"{study_name} :: {variant.label} :: seed={seed}",
        command=cmd,
        cwd=PROJECT_ROOT,
    )
