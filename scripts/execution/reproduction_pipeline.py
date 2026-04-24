#!/usr/bin/env python3
"""
Cross-platform reproduction pipeline for SPECTRA publication runs.

The NYUv2 publication path supports:
  - optional dataset preparation via ``nyuv2_generate``
  - full method-matrix execution
  - seed sweeps with deterministic override injection
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.execution.experiment_runner import METHODS_BY_DATASET

NYUV2_PUBLICATION_METHODS = list(METHODS_BY_DATASET["nyuv2"])
DEFAULT_NYUV2_SEEDS = [42, 43, 44]


def _current_timestamp() -> datetime:
    return datetime.now().astimezone()


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.isoformat(sep=" ", timespec="seconds")


def _format_elapsed(elapsed_seconds: float) -> str:
    return f"{elapsed_seconds:.3f}s"


def _parse_csv_list(raw_value: str | None) -> List[str]:
    if not raw_value:
        return []
    parts = [part.strip() for part in raw_value.split(",")]
    return [part for part in parts if part]


def resolve_nyuv2_methods(raw_methods: str | None = None) -> List[str]:
    if raw_methods is None or not raw_methods.strip():
        return list(NYUV2_PUBLICATION_METHODS)

    methods = _parse_csv_list(raw_methods)
    invalid = [method for method in methods if method not in NYUV2_PUBLICATION_METHODS]
    if invalid:
        raise ValueError(
            f"Unknown NYUv2 methods: {invalid}. Valid choices: {NYUV2_PUBLICATION_METHODS}"
        )
    return methods


def resolve_seed_sweep(raw_seeds: Sequence[int] | None = None) -> List[int]:
    if not raw_seeds:
        return list(DEFAULT_NYUV2_SEEDS)
    return [int(seed) for seed in raw_seeds]


def build_nyuv2_publication_plan(
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    include_data_prep: bool = True,
    data_prep_args: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    resolved_methods = list(methods or NYUV2_PUBLICATION_METHODS)
    resolved_seeds = list(seeds or DEFAULT_NYUV2_SEEDS)
    plan: List[Dict[str, object]] = []

    if include_data_prep:
        plan.append(
            {
                "label": "Prepare NYUv2 dataset",
                "experiment": "nyuv2_generate",
                "args": list(data_prep_args or []),
                "seed": None,
                "kind": "data",
            }
        )

    for seed in resolved_seeds:
        for method in resolved_methods:
            plan.append(
                {
                    "label": f"Run {method} on NYUv2 (seed={seed})",
                    "experiment": f"{method}_nyuv2",
                    "args": [f"seed={seed}"],
                    "seed": seed,
                    "kind": "train",
                }
            )

    return plan


def run_experiment(
    experiment: str,
    hydra_args: List[str] | None = None,
    epochs: int | None = None,
    dry_run: bool = False,
) -> int:
    if hydra_args is None:
        hydra_args = []

    runner_script = PROJECT_ROOT / "scripts" / "execution" / "experiment_runner.py"
    cmd = [sys.executable, str(runner_script), experiment]
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    cmd.extend(hydra_args)

    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return 0

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as exc:
        print(f"Error running experiment '{experiment}': {exc}")
        return exc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publication reproduction pipeline for SPECTRA"
    )
    parser.add_argument(
        "--pipeline",
        choices=["nyuv2_publication"],
        default="nyuv2_publication",
        help="Named pipeline to execute",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Resume from a stage label, experiment alias, or '<experiment>@<seed>' token",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without executing them",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs for Hydra training experiments",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip the NYUv2 dataset preparation stage",
    )
    parser.add_argument(
        "--force-data-prep",
        action="store_true",
        help="Pass --force to nyuv2_generate",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="Pass --skip-validation to nyuv2_generate",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Pass --keep-staging to nyuv2_generate",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated subset of NYUv2 methods to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seed sweep for NYUv2 publication runs (default: 42 43 44)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failures instead of stopping at the first failed stage",
    )

    args, extra_hydra_args = parser.parse_known_args()

    methods = resolve_nyuv2_methods(args.methods)
    seeds = resolve_seed_sweep(args.seeds)

    data_prep_args: List[str] = []
    if args.force_data_prep:
        data_prep_args.append("--force")
    if args.skip_data_validation:
        data_prep_args.append("--skip-validation")
    if args.keep_staging:
        data_prep_args.append("--keep-staging")

    if args.pipeline != "nyuv2_publication":
        raise ValueError(f"Unsupported pipeline: {args.pipeline}")

    plan = build_nyuv2_publication_plan(
        methods=methods,
        seeds=seeds,
        include_data_prep=not args.skip_data_prep,
        data_prep_args=data_prep_args,
    )

    if args.start_from:
        matched_index = None
        for idx, stage in enumerate(plan):
            token = stage["experiment"]
            if stage["seed"] is not None:
                token = f"{token}@{stage['seed']}"
            if args.start_from in {str(stage["label"]), str(stage["experiment"]), token}:
                matched_index = idx
                break

        if matched_index is None:
            valid_tokens = []
            for stage in plan:
                valid_tokens.append(str(stage["experiment"]))
                if stage["seed"] is not None:
                    valid_tokens.append(f"{stage['experiment']}@{stage['seed']}")
            print(f"Error: Unknown start stage '{args.start_from}'")
            print(f"Valid options include: {', '.join(valid_tokens)}")
            return 1

        plan = plan[matched_index:]

    pipeline_start = _current_timestamp()
    pipeline_start_perf = perf_counter()

    print("=" * 58)
    print("  SPECTRA NYUv2 Publication Pipeline")
    print("=" * 58)
    print(f"  Pipeline Started At: {_format_timestamp(pipeline_start)}")
    print(f"  Methods:             {', '.join(methods)}")
    print(f"  Seeds:               {', '.join(str(seed) for seed in seeds)}")
    if args.epochs:
        print(f"  Epoch Override:      {args.epochs}")
    if args.dry_run:
        print("  Mode:                DRY-RUN")
    if args.skip_data_prep:
        print("  Data Prep:           skipped")
    print("\n[Initiating Pipeline...]\n")

    failed_stages: List[str] = []
    pipeline_status = "unknown"
    exit_code = 1

    try:
        total = len(plan)
        for index, stage in enumerate(plan, start=1):
            label = str(stage["label"])
            experiment = str(stage["experiment"])
            stage_args = list(stage["args"])
            if stage["kind"] == "train":
                stage_args.extend(extra_hydra_args)

            print(f"\n[{index}/{total}] {label}")
            print(f"[handoff] Launching {experiment}")
            step_start = _current_timestamp()
            step_start_perf = perf_counter()
            print(f"  -> [{experiment}] Started at {_format_timestamp(step_start)}")

            result = run_experiment(
                experiment,
                hydra_args=stage_args,
                epochs=args.epochs if stage["kind"] == "train" else None,
                dry_run=args.dry_run,
            )

            step_end = _current_timestamp()
            elapsed_seconds = perf_counter() - step_start_perf
            step_status = "completed" if result == 0 else f"failed (exit code {result})"
            print(f"  -> [{experiment}] Ended at {_format_timestamp(step_end)}")
            print(f"  -> [{experiment}] Status: {step_status}")
            print(f"  -> [{experiment}] Elapsed: {_format_elapsed(elapsed_seconds)}")

            if result != 0:
                failed_stages.append(label)
                if not args.keep_going:
                    print("Stopping pipeline after failure.")
                    break

        pipeline_status = "completed" if not failed_stages else "completed with failures"
        exit_code = 1 if failed_stages else 0
        return exit_code

    except KeyboardInterrupt:
        pipeline_status = "interrupted by user"
        exit_code = 130
        print("\nPipeline interrupted by user.")
        return exit_code

    finally:
        print("\n" + "=" * 58)
        if failed_stages:
            print(f"  Pipeline completed with {len(failed_stages)} failed stage(s):")
            for stage in failed_stages:
                print(f"    - {stage}")
        elif pipeline_status == "interrupted by user":
            print("  Pipeline interrupted before completion.")
        else:
            print("  Pipeline fully complete.")

        pipeline_end = _current_timestamp()
        pipeline_elapsed_seconds = perf_counter() - pipeline_start_perf
        print(f"  Pipeline Status:  {pipeline_status}")
        print(f"  Pipeline Ended:   {_format_timestamp(pipeline_end)}")
        print(f"  Total Runtime:    {_format_elapsed(pipeline_elapsed_seconds)}")
        print("  Review 'outputs/' for logs and checkpoints.")
        print("=" * 58)


if __name__ == "__main__":
    sys.exit(main())
