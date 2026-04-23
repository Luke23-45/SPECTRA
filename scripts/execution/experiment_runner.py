#!/usr/bin/env python3
"""
Cross-platform execution script for SPECTRA experiments.

Supports:
  - Hydra training runs through ``scripts.run_training``
  - Standalone utility scripts such as NYUv2 dataset generation
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HYDRA_TRAINING_MODULE = "scripts.run_training"

METHODS_BY_DATASET: Dict[str, List[str]] = {
    "synthetic": ["static", "kendall", "uwso", "gradnorm_proxy", "pcgrad", "bpgs"],
    "nyuv2": ["static", "kendall", "uwso", "gradnorm_proxy", "pcgrad", "bpgs", "bpgs_alb"],
    "clinical": ["bpgs"],
}

SCRIPT_EXPERIMENTS: Dict[str, Dict[str, object]] = {
    "nyuv2_generate": {
        "type": "script",
        "module": "scripts/data/nyuv2_generate.py",
    }
}


def _hydra_alias(dataset: str, method: str) -> str:
    if dataset == "synthetic":
        return method
    return f"{method}_{dataset}"


def _hydra_defaults(dataset: str, method: str) -> List[str]:
    return [f"dataset={dataset}", f"method={method}"]


def build_experiment_registry() -> Dict[str, Dict[str, object]]:
    experiments: Dict[str, Dict[str, object]] = {}

    for dataset, methods in METHODS_BY_DATASET.items():
        for method in methods:
            alias = _hydra_alias(dataset, method)
            experiments[alias] = {
                "type": "hydra",
                "module": HYDRA_TRAINING_MODULE,
                "dataset": dataset,
                "method": method,
                "defaults": _hydra_defaults(dataset, method),
            }

    experiments.update(SCRIPT_EXPERIMENTS)
    return experiments


EXPERIMENTS = build_experiment_registry()


def grouped_experiments() -> List[Tuple[str, List[str]]]:
    return [
        ("Data Preparation", ["nyuv2_generate"]),
        ("Training - Synthetic", [_hydra_alias("synthetic", m) for m in METHODS_BY_DATASET["synthetic"]]),
        ("Training - NYUv2", [_hydra_alias("nyuv2", m) for m in METHODS_BY_DATASET["nyuv2"]]),
        ("Training - Clinical", [_hydra_alias("clinical", m) for m in METHODS_BY_DATASET["clinical"]]),
    ]


def _current_timestamp() -> datetime:
    return datetime.now().astimezone()


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.isoformat(sep=" ", timespec="seconds")


def _format_elapsed(elapsed_seconds: float) -> str:
    return f"{elapsed_seconds:.3f}s"


def _has_override(args: List[str], key: str) -> bool:
    key_prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(arg.startswith(key_prefixes) for arg in args)


def default_overrides_for_experiment(
    experiment: str,
    existing_overrides: List[str] | None = None,
) -> List[str]:
    existing = list(existing_overrides or [])
    entry = EXPERIMENTS[experiment]
    defaults = entry.get("defaults", [])
    result: List[str] = []

    for override in defaults:
        key = str(override).split("=")[0]
        if not _has_override(existing, key):
            result.append(str(override))

    return result


def build_command(
    experiment: str,
    extra_args: List[str] | None = None,
    epochs: int | None = None,
) -> List[str]:
    extra_args = list(extra_args or [])
    entry = EXPERIMENTS[experiment]

    if entry["type"] == "hydra":
        resolved_overrides = extra_args + default_overrides_for_experiment(experiment, extra_args)
        if epochs is not None and not _has_override(resolved_overrides, "train.epochs"):
            resolved_overrides.append(f"train.epochs={epochs}")
        return [sys.executable, "-m", str(entry["module"])] + resolved_overrides

    script_path = PROJECT_ROOT / str(entry["module"])
    return [sys.executable, str(script_path)] + extra_args


def print_usage() -> None:
    print("Usage: python scripts/execution/experiment_runner.py [--epochs N] <experiment> [args...]")
    print("")
    print("Available experiments:")

    for phase_name, aliases in grouped_experiments():
        print(f"\n  [{phase_name}]")
        for alias in aliases:
            entry = EXPERIMENTS[alias]
            type_tag = "hydra" if entry["type"] == "hydra" else "script"
            print(f"    {alias:<24} ({type_tag})  {entry['module']}")

    print("")
    print("Examples:")
    print("  python scripts/execution/experiment_runner.py nyuv2_generate --force")
    print("  python scripts/execution/experiment_runner.py bpgs_nyuv2 train.epochs=200")
    print("  python scripts/execution/experiment_runner.py kendall_nyuv2 train.batch_size=4")
    print("  python scripts/execution/experiment_runner.py bpgs_clinical")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform execution script for SPECTRA experiments",
        add_help=False,
    )
    parser.add_argument("experiment", nargs="?", help="Experiment alias to run")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs for Hydra training runs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiment aliases and exit",
    )

    args, extra_args = parser.parse_known_args()

    if args.list:
        print_usage()
        return 0

    if not args.experiment or args.experiment in ["-h", "--help", "help"]:
        print_usage()
        return 0 if args.experiment in ["-h", "--help", "help"] else 1

    experiment = args.experiment.lower()
    if experiment not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment}'")
        print(f"Valid options are: {', '.join(sorted(EXPERIMENTS.keys()))}")
        return 1

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    path_parts = [str(PROJECT_ROOT)]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)

    cmd = build_command(experiment, extra_args=extra_args, epochs=args.epochs)
    entry = EXPERIMENTS[experiment]
    exp_type = str(entry["type"])

    start_time = _current_timestamp()
    start_perf = perf_counter()

    print("=" * 58)
    print(f" Starting Experiment: {experiment}")
    print(f" Type:                {exp_type}")
    print(f" Start Time:          {_format_timestamp(start_time)}")
    if exp_type == "hydra":
        print(f" Hydra Overrides:     {' '.join(cmd[3:])}")
    else:
        print(f" Script:              {entry['module']}")
        if len(cmd) > 2:
            print(f" Script Args:         {' '.join(cmd[2:])}")
    print("=" * 58)

    os.chdir(PROJECT_ROOT)

    status = "unknown"
    return_code = 1

    try:
        result = subprocess.run(cmd, env=env)
        return_code = result.returncode
        status = "completed" if return_code == 0 else f"failed (exit code {return_code})"
        return return_code
    except KeyboardInterrupt:
        status = "interrupted by user"
        print("\nExperiment interrupted by user")
        return_code = 130
        return 130
    except Exception as exc:
        status = f"crashed ({type(exc).__name__})"
        print(f"Error running experiment: {exc}")
        return_code = 1
        return 1
    finally:
        end_time = _current_timestamp()
        elapsed_seconds = perf_counter() - start_perf
        print(f"\n[Experiment '{experiment}' Finished]")
        print(f"  Status:  {status}")
        print(f"  End:     {_format_timestamp(end_time)}")
        print(f"  Elapsed: {_format_elapsed(elapsed_seconds)}")


if __name__ == "__main__":
    sys.exit(main())
