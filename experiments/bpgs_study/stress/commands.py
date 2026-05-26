from __future__ import annotations

import sys
from pathlib import Path

from experiments.bpgs_study.common.paths import OUTPUT_ROOT, PROJECT_ROOT
from experiments.bpgs_study.common.runtime import Stage
from experiments.bpgs_study.common.specs import EmpiricalVariant, OverrideSpec


def _resolve_output_dir(study_name: str, variant: EmpiricalVariant, method: str, seed: int) -> Path:
    return OUTPUT_ROOT / study_name / variant.label / method / f"seed_{seed}"


def build_empirical_stage(
    study_name: str,
    variant: EmpiricalVariant,
    method: str,
    seed: int,
    overrides: tuple[OverrideSpec, ...],
) -> Stage:
    """Build a Hydra empirical experiment command for a single variant-method-seed triple.

    The stress runner owns this logic entirely.  Overrides are
    method-scoped: only overrides whose ``scope_method`` matches
    ``method`` (or have no scope) are applied.
    """
    from experiments.bpgs_study.common.runtime import filter_overrides_for_method

    applicable = filter_overrides_for_method(overrides, method)

    cmd = [sys.executable, "-m", "scripts.run_empirical_experiment"]
    cmd.append(f"experiment={variant.experiment}")
    cmd.append(f"family={variant.family}")
    cmd.append(f"methods.names=[{method}]")
    cmd.append(f"seeds.values=[{seed}]")
    cmd.append("runtime.device=cuda")

    for override in variant.overrides:
        cmd.append(str(override))

    for override in applicable:
        cmd.append(override.to_hydra_arg())

    output_dir = _resolve_output_dir(study_name, variant, method, seed)
    cmd.append(f"output.root_dir={OUTPUT_ROOT / study_name}")
    cmd.append(f"output.run_dir={output_dir}")
    cmd.append(f"hydra.run.dir={output_dir}")

    return Stage(
        label=f"{study_name} :: {variant.label} :: {method} :: seed={seed}",
        command=cmd,
        cwd=PROJECT_ROOT,
    )
