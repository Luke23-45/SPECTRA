from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.bpgs_study.aggregation_v2 import aggregate_study_full, write_csv
from analysis.bpgs_study.cross_study import write_cross_study_report
from analysis.bpgs_study.paths import ANALYSIS_OUTPUT_ROOT
from analysis.bpgs_study.publication import write_publication_artifacts
from analysis.bpgs_study.reproducibility import (
    write_cross_study_manifest,
    write_study_manifest,
)
from studies.bpgs_study.definitions import STUDIES, get_study


def _aggregate_single(study_name: str, confidence: float, baseline_variant: str) -> int:
    study = get_study(study_name)
    aggregated = aggregate_study_full(study, confidence, baseline_variant)

    output_dir = ANALYSIS_OUTPUT_ROOT / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write all CSV data
    for key in (
        "records", "summary", "epoch_curves", "weight_trajectories",
        "gradient_diagnostics", "per_task_losses", "statistical_tests",
        "confidence_intervals",
    ):
        data = aggregated.get(key, [])
        if data:
            write_csv(output_dir / f"{key}.csv", data)

    # Write full JSON
    (output_dir / "full_aggregation.json").write_text(
        json.dumps(aggregated, indent=2, default=str), encoding="utf-8"
    )

    # Write publication artifacts (LaTeX tables, figure data)
    write_publication_artifacts(study_name, study.kind, aggregated, output_dir)

    # Write reproducibility manifest
    write_study_manifest(study, output_dir)

    print(f"Wrote full study report to {output_dir}")
    print(f"  records:          {len(aggregated.get('records', []))}")
    print(f"  epoch_curves:     {len(aggregated.get('epoch_curves', []))}")
    print(f"  weight_traj:      {len(aggregated.get('weight_trajectories', []))}")
    print(f"  grad_diag:        {len(aggregated.get('gradient_diagnostics', []))}")
    print(f"  per_task_losses:  {len(aggregated.get('per_task_losses', []))}")
    print(f"  stat_tests:       {len(aggregated.get('statistical_tests', []))}")
    print(f"  ci_rows:          {len(aggregated.get('confidence_intervals', []))}")
    return 0


def _aggregate_cross(confidence: float, baseline_variant: str, groups: list[str] | None) -> int:
    studies = list(STUDIES)
    if groups:
        studies = [s for s in studies if s.group in groups]
    if not studies:
        print("No studies matched the specified groups.")
        return 1

    output_dir = ANALYSIS_OUTPUT_ROOT / "cross_study"
    write_cross_study_report(studies, output_dir, confidence, baseline_variant)
    write_cross_study_manifest(studies, output_dir)

    print(f"Wrote cross-study report to {output_dir}")
    print(f"  studies: {len(studies)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate BPGS study outputs into dedicated analysis artifacts."
    )
    parser.add_argument("--study", help="Study name to aggregate.")
    parser.add_argument(
        "--cross-study", action="store_true",
        help="Aggregate all studies into a unified cross-study report.",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Filter cross-study to these groups (ablation, stress, real_data, optional).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level for intervals (default: 0.95).",
    )
    parser.add_argument(
        "--baseline-variant", type=str, default="static",
        help="Baseline variant for statistical tests (default: static).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List study names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        current_group = None
        for study in STUDIES:
            if study.group != current_group:
                current_group = study.group
                print(f"\n[{current_group}]")
            print(f"  {study.name} :: {study.description}")
        return 0

    if args.cross_study:
        return _aggregate_cross(args.confidence, args.baseline_variant, args.groups)

    if not args.study:
        print("--study is required unless --list or --cross-study is used.")
        return 1

    return _aggregate_single(args.study, args.confidence, args.baseline_variant)


if __name__ == "__main__":
    raise SystemExit(main())
