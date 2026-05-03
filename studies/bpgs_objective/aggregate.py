from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectra.evaluation.metrics import compute_delta_m
from studies.bpgs_objective.definitions import OUTPUT_ROOT, get_study


REPORT_ROOT = OUTPUT_ROOT / "reports"


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_float(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _metrics_rows(metrics_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            val_total_loss = _parse_float(row.get("val/total_loss"))
            epoch_value = _parse_float(row.get("epoch"))
            if val_total_loss is None or epoch_value is None:
                continue
            rows.append(
                {
                    "epoch": int(epoch_value),
                    "val/total_loss": val_total_loss,
                    "val/miou": _parse_float(row.get("val/miou")),
                    "val/depth_abs_rel": _parse_float(row.get("val/depth_abs_rel")),
                    "val/normals_mean_angle": _parse_float(row.get("val/normals_mean_angle")),
                }
            )
    return rows


def _selected_epoch(summary: Dict[str, object]) -> Optional[int]:
    selected = summary.get("selected_checkpoint", {})
    if not isinstance(selected, dict):
        return None
    epoch = selected.get("epoch")
    return int(epoch) if epoch is not None else None


def _select_row(rows: List[Dict[str, float]], epoch: Optional[int]) -> Dict[str, float]:
    if epoch is not None:
        for row in rows:
            if int(row["epoch"]) == int(epoch):
                return row
    return min(rows, key=lambda row: row["val/total_loss"])


def _metric_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _metric_std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_training(study_name: str) -> Dict[str, object]:
    study_root = OUTPUT_ROOT / study_name
    records: List[Dict[str, object]] = []

    if not study_root.exists():
        return {"records": [], "summary": []}

    for run_dir in study_root.glob("*/*"):
        metadata_path = run_dir / "metadata.json"
        metrics_candidates = list((run_dir / "csv_logs").glob("*/*.csv"))
        if not metadata_path.exists() or not metrics_candidates:
            continue

        metadata = _read_json(metadata_path)
        run_summary = _read_json(run_dir / "run_summary.json") if (run_dir / "run_summary.json").exists() else {}
        rows = _metrics_rows(metrics_candidates[0])
        if not rows:
            continue

        row = _select_row(rows, _selected_epoch(run_summary))
        variant = run_dir.parent.name
        seed_match = re.search(r"seed_(\d+)$", run_dir.name)
        seed = int(seed_match.group(1)) if seed_match else None
        record = {
            "study": study_name,
            "variant": variant,
            "training_seed": seed,
            "artifact_dir": str(run_dir),
            "method": metadata["experiment"]["method"],
            "dataset": metadata["experiment"]["dataset"],
            "subset_id": metadata.get("dataset", {}).get("train_subset_id"),
            "subset_budget": metadata.get("dataset", {}).get("subset_pct"),
            "subset_seed": metadata.get("dataset", {}).get("subset_seed"),
            "selected_checkpoint": (run_summary.get("selected_checkpoint", {}) or {}).get("path"),
            "selected_epoch": row["epoch"],
            "stopped_epoch": run_summary.get("stopped_epoch"),
            "runtime_seconds": run_summary.get("elapsed_seconds"),
            "miou": row.get("val/miou"),
            "abs_rel": row.get("val/depth_abs_rel"),
            "mean_angle_deg": row.get("val/normals_mean_angle"),
            "val_total_loss": row.get("val/total_loss"),
        }
        records.append(record)

    baselines = {int(r["training_seed"]): r for r in records if r["variant"] == "static" and r["training_seed"] is not None}
    for record in records:
        seed = record.get("training_seed")
        baseline = baselines.get(int(seed)) if seed is not None else None
        if baseline and all(record.get(k) is not None for k in ("miou", "abs_rel", "mean_angle_deg")):
            record["delta_m"] = compute_delta_m(
                results={"miou": float(record["miou"]), "abs_rel": float(record["abs_rel"]), "mean_angle_deg": float(record["mean_angle_deg"])},
                baseline={"miou": float(baseline["miou"]), "abs_rel": float(baseline["abs_rel"]), "mean_angle_deg": float(baseline["mean_angle_deg"])},
                metric_configs={"miou": False, "abs_rel": True, "mean_angle_deg": True},
            )
        else:
            record["delta_m"] = float("nan")

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["variant"])].append(record)

    summary: List[Dict[str, object]] = []
    for variant, variant_records in sorted(grouped.items()):
        metrics = {
            "miou": [float(r["miou"]) for r in variant_records if r.get("miou") is not None],
            "abs_rel": [float(r["abs_rel"]) for r in variant_records if r.get("abs_rel") is not None],
            "mean_angle_deg": [float(r["mean_angle_deg"]) for r in variant_records if r.get("mean_angle_deg") is not None],
            "delta_m": [float(r["delta_m"]) for r in variant_records if not math.isnan(float(r["delta_m"]))],
            "runtime_seconds": [float(r["runtime_seconds"]) for r in variant_records if r.get("runtime_seconds") is not None],
            "stopped_epoch": [float(r["stopped_epoch"]) for r in variant_records if r.get("stopped_epoch") is not None],
        }
        summary.append(
            {
                "study": study_name,
                "variant": variant,
                "method": variant_records[0]["method"],
                "dataset": variant_records[0]["dataset"],
                "n_runs": len(variant_records),
                "miou_mean": _metric_mean(metrics["miou"]),
                "miou_std": _metric_std(metrics["miou"]),
                "abs_rel_mean": _metric_mean(metrics["abs_rel"]),
                "abs_rel_std": _metric_std(metrics["abs_rel"]),
                "mean_angle_deg_mean": _metric_mean(metrics["mean_angle_deg"]),
                "mean_angle_deg_std": _metric_std(metrics["mean_angle_deg"]),
                "delta_m_mean": _metric_mean(metrics["delta_m"]) if metrics["delta_m"] else float("nan"),
                "delta_m_std": _metric_std(metrics["delta_m"]) if metrics["delta_m"] else float("nan"),
                "runtime_seconds_mean": _metric_mean(metrics["runtime_seconds"]),
                "runtime_seconds_std": _metric_std(metrics["runtime_seconds"]),
                "stopped_epoch_mean": _metric_mean(metrics["stopped_epoch"]),
                "stopped_epoch_std": _metric_std(metrics["stopped_epoch"]),
            }
        )

    return {"records": records, "summary": summary}


def _aggregate_empirical(study_name: str) -> Dict[str, object]:
    study_root = OUTPUT_ROOT / study_name
    records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    if not study_root.exists():
        return {"records": [], "summary": []}

    for results_path in study_root.glob("*/*/metrics/results.csv"):
        variant = results_path.parents[2].name
        seed_match = re.search(r"seed_(\d+)$", results_path.parents[1].name)
        seed = int(seed_match.group(1)) if seed_match else None
        with results_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["study"] = study_name
                row["variant"] = variant
                row["empirical_seed"] = seed
                records.append(dict(row))

    for summary_path in study_root.glob("*/*/metrics/summary.csv"):
        variant = summary_path.parents[2].name
        with summary_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["study"] = study_name
                row["variant"] = variant
                summary_rows.append(dict(row))

    return {"records": records, "summary": summary_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate one isolated BPGS objective study.")
    parser.add_argument("--study", required=True)
    args = parser.parse_args()

    study = get_study(args.study)
    aggregated = _aggregate_training(args.study) if study.kind == "training" else _aggregate_empirical(args.study)
    output_dir = REPORT_ROOT / args.study
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "tidy_runs.csv", aggregated["records"])
    _write_csv(output_dir / "summary.csv", aggregated["summary"])
    (output_dir / "summary.json").write_text(json.dumps(aggregated, indent=2, default=str), encoding="utf-8")

    print(f"Wrote isolated study report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
