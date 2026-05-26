from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bpgs_study.common.catalog import get_registered_studies, get_study_entry
from experiments.bpgs_study.common.runtime import RunRequest
from experiments.bpgs_study.common.specs import OverrideSpec
from experiments.bpgs_study.common.validation import parse_raw_overrides

STUDIES = get_registered_studies()


def _print_study_listing() -> None:
    current_group = None
    for study in STUDIES:
        if study.group != current_group:
            current_group = study.group
            print(f"\n[{current_group}]")
        print(f"  {study.name} :: {study.description}")
        print(f"    config: {study.config_path.as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reliable, study-owned execution plans for the BPGS study suite."
    )
    parser.add_argument("--study", help="Study name to execute.")
    parser.add_argument("--list", action="store_true", help="List study names and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--start-from", type=int, default=1, help="1-based stage index to resume from.")
    parser.add_argument("--keep-going", action="store_true", help="Continue remaining stages after failures.")
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip preparatory stages such as NYUv2 subset generation.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant label to run. Repeat to select multiple variants within the study.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Seed to run. Repeat to select multiple seeds from the study definition.",
    )
    parser.add_argument(
        "--set",
        dest="raw_overrides",
        action="append",
        default=[],
        help=(
            "Runtime override in key=value or method:key=value form. "
            "Use 'method:key=value' to scope an override to only variants "
            "whose method matches (e.g. bpgs:method.s_mode=stateless). "
            "Repeat to apply multiple overrides."
        ),
    )
    args = parser.parse_args()

    if args.list:
        _print_study_listing()
        return 0

    if not args.study:
        print("--study is required unless --list is used.")
        return 1

    try:
        overrides = parse_raw_overrides(tuple(args.raw_overrides))
        entry = get_study_entry(args.study)
        request = RunRequest(
            variant_labels=tuple(args.variant),
            seeds=tuple(args.seed),
            overrides=overrides,
            skip_prep=args.skip_prep,
        )
        stages = entry.plan(request)
    except (KeyError, TypeError, ValueError) as exc:
        print(str(exc))
        return 1

    if not stages:
        print(f"No stages were selected for study '{entry.spec.name}'.")
        return 1

    for idx, stage in enumerate(stages, start=1):
        if idx < max(1, args.start_from):
            continue
        print(f"[{idx}/{len(stages)}] {stage.label}")
        if args.dry_run:
            print(f"[DRY-RUN] cwd={stage.cwd}")
            print(f"[DRY-RUN] {' '.join(stage.command)}")
            continue
        result = subprocess.run(stage.command, cwd=stage.cwd).returncode
        if result != 0:
            print(f"Stage failed with exit code {result}: {stage.label}")
            if not args.keep_going:
                return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
