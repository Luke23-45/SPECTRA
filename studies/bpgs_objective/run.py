from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.bpgs_objective.definitions import OUTPUT_ROOT, SUBSET_ROOT, EmpiricalVariant, STUDIES, TrainingVariant, get_study, nyuv2_subset_id, nyuv2_subset_path


TRAIN_MODULE = "scripts.run_training"
EMPIRICAL_MODULE = "scripts.run_empirical_experiment"
SUBSET_SCRIPT = Path(__file__).resolve().parent / "nyuv2_subsets.py"


@dataclass(frozen=True)
class Stage:
    label: str
    command: List[str]


def _python_module(module: str, overrides: List[str]) -> List[str]:
    return [sys.executable, "-m", module, *overrides]


def _training_output_dir(study_name: str, variant_label: str, seed: int) -> str:
    return f"./outputs/studies/bpgs_objective/{study_name}/{variant_label}/seed_{seed}"


def _empirical_output_dir(study_name: str, variant_label: str, seed: int) -> str:
    return f"./outputs/studies/bpgs_objective/{study_name}/{variant_label}/seed_{seed}"


def _build_training_stages(study_name: str, variant: TrainingVariant) -> List[Stage]:
    stages: List[Stage] = []
    for seed in variant.seeds:
        run_name = f"{study_name}_{variant.label}_s{seed}"
        overrides = [
            f"dataset={variant.dataset}",
            f"method={variant.method}",
            f"seed={seed}",
            f"run_name={run_name}",
            f"output_dir={_training_output_dir(study_name, variant.label, int(seed))}",
            f"train.epochs={variant.epochs}",
            "train.save_ckpt=true",
            "train.selection_metric=val/total_loss",
            "train.selection_mode=min",
            f"train.early_stop={'true' if variant.early_stop else 'false'}",
        ]
        if variant.early_stop_patience is not None:
            overrides.append(f"train.early_stop_patience={variant.early_stop_patience}")
        if variant.early_stop_min_delta is not None:
            overrides.append(f"train.early_stop_min_delta={variant.early_stop_min_delta}")
        if variant.use_subset_file and variant.subset_budget and variant.subset_seed is not None:
            subset_file = nyuv2_subset_path(variant.subset_budget, variant.subset_seed)
            overrides.extend(
                [
                    f"train_subset_file={subset_file.as_posix()}",
                    f"train_subset_id={nyuv2_subset_id(variant.subset_budget, variant.subset_seed)}",
                    f"subset_pct={float(int(variant.subset_budget)) / 100.0}",
                    f"subset_seed={variant.subset_seed}",
                ]
            )
        overrides.extend(list(variant.extra_overrides))
        stages.append(Stage(label=f"{study_name}:{variant.label}:seed={seed}", command=_python_module(TRAIN_MODULE, overrides)))
    return stages


def _build_empirical_stages(study_name: str, variant: EmpiricalVariant) -> List[Stage]:
    stages: List[Stage] = []
    for seed in variant.seeds:
        overrides = [
            f"experiment={variant.experiment}",
            f"family={variant.family}",
            f"methods.names=[{','.join(variant.methods)}]",
            f"seeds.values=[{seed}]",
            f"output.run_dir={_empirical_output_dir(study_name, variant.label, int(seed))}",
            *variant.overrides,
            f"experiment.generator.seed={seed}",
        ]
        stages.append(Stage(label=f"{study_name}:{variant.label}:seed={seed}", command=_python_module(EMPIRICAL_MODULE, overrides)))
    return stages


def build_stages(study_name: str) -> List[Stage]:
    study = get_study(study_name)
    stages: List[Stage] = []
    if study.requires_nyuv2_subsets:
        stages.append(Stage(label=f"{study_name}:prepare_subsets", command=[sys.executable, str(SUBSET_SCRIPT)]))

    for variant in study.variants:
        if isinstance(variant, TrainingVariant):
            stages.extend(_build_training_stages(study_name, variant))
        else:
            stages.extend(_build_empirical_stages(study_name, variant))
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated studies for the BPGS objective paper.")
    parser.add_argument("--study", help="Study name to execute.")
    parser.add_argument("--list", action="store_true", help="List study names and exit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.list:
        for study in STUDIES:
            print(f"{study.name} :: {study.description}")
        return 0

    if not args.study:
        print("--study is required unless --list is used.")
        return 1

    stages = build_stages(args.study)
    for idx, stage in enumerate(stages, start=1):
        if idx < max(1, args.start_from):
            continue
        print(f"[{idx}/{len(stages)}] {stage.label}")
        if args.dry_run:
            print(f"[DRY-RUN] {' '.join(stage.command)}")
            continue
        result = subprocess.run(stage.command, cwd=PROJECT_ROOT).returncode
        if result != 0:
            print(f"Stage failed with exit code {result}: {stage.label}")
            if not args.keep_going:
                return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
