from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from spectra.evaluation.metrics import compute_delta_m
from studies.bpgs_study.common.paths import resolve_study_output_root
from studies.bpgs_study.common.specs import StudySpec
from analysis.bpgs_study.statistics import (
    compute_confidence_intervals,
    confidence_interval,
    pairwise_tests,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_float(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _metric_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _metric_std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _discover_csv_columns(metrics_path: Path) -> List[str]:
    """Read header row to discover all available columns."""
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _read_all_csv_rows(metrics_path: Path) -> List[Dict[str, str]]:
    """Read all rows from a CSV as raw string dicts."""
    rows: List[Dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

VAL_METRIC_PREFIXES = ("val/",)
TRAIN_LOSS_PREFIXES = ("train/",)
BPGS_PREFIXES = ("bpgs/",)
HEALTH_PREFIXES = ("health/",)
LR_PREFIXES = ("lr",)


def _classify_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Group CSV columns by semantic category."""
    classified: Dict[str, List[str]] = {
        "epoch": [],
        "val_metrics": [],
        "train_losses": [],
        "bpgs_weights": [],
        "gradient_health": [],
        "lr": [],
        "other": [],
    }
    for col in columns:
        if col == "epoch":
            classified["epoch"].append(col)
        elif col.startswith(VAL_METRIC_PREFIXES):
            classified["val_metrics"].append(col)
        elif col.startswith(BPGS_PREFIXES):
            classified["bpgs_weights"].append(col)
        elif col.startswith(HEALTH_PREFIXES):
            classified["gradient_health"].append(col)
        elif col.startswith(LR_PREFIXES):
            classified["lr"].append(col)
        elif col.startswith(TRAIN_LOSS_PREFIXES):
            classified["train_losses"].append(col)
        else:
            classified["other"].append(col)
    return classified


# ---------------------------------------------------------------------------
# Core data extraction
# ---------------------------------------------------------------------------

def _extract_run_info(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract run identity and metadata from a training run directory."""
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    metadata = _read_json(metadata_path)
    run_summary = {}
    rs_path = run_dir / "run_summary.json"
    if rs_path.exists():
        run_summary = _read_json(rs_path)

    variant = run_dir.parent.name
    seed_match = re.search(r"seed_(\d+)$", run_dir.name)
    seed = int(seed_match.group(1)) if seed_match else None

    return {
        "study": None,
        "variant": variant,
        "training_seed": seed,
        "artifact_dir": str(run_dir),
        "method": metadata.get("experiment", {}).get("method"),
        "dataset": metadata.get("experiment", {}).get("dataset"),
        "run_name": metadata.get("experiment", {}).get("run_name"),
        "run_id": metadata.get("experiment", {}).get("run_id"),
        "subset_id": metadata.get("dataset", {}).get("train_subset_id"),
        "subset_budget": metadata.get("dataset", {}).get("subset_pct"),
        "subset_seed": metadata.get("dataset", {}).get("subset_seed"),
        "selected_checkpoint": (run_summary.get("selected_checkpoint", {}) or {}).get("path"),
        "stopped_epoch": run_summary.get("stopped_epoch"),
        "global_step": run_summary.get("global_step"),
        "runtime_seconds": run_summary.get("elapsed_seconds"),
        "fit_started_at": run_summary.get("fit_started_at"),
        "fit_ended_at": run_summary.get("fit_ended_at"),
        "system_info": metadata.get("system"),
        "git_info": metadata.get("git"),
        "training_config": metadata.get("training"),
        "tasks": metadata.get("tasks"),
    }


def _extract_epoch_curves(
    metrics_path: Path,
    run_info: Dict[str, Any],
    study_name: str,
) -> List[Dict[str, object]]:
    """Extract ALL epoch-level data with run identity."""
    raw_rows = _read_all_csv_rows(metrics_path)
    if not raw_rows:
        return []

    columns = list(raw_rows[0].keys())
    classified = _classify_columns(columns)

    epoch_rows: List[Dict[str, object]] = []
    for raw in raw_rows:
        epoch_val = _parse_float(raw.get("epoch"))
        if epoch_val is None:
            continue
        row: Dict[str, object] = {
            "study": study_name,
            "variant": run_info["variant"],
            "seed": run_info["training_seed"],
            "method": run_info["method"],
            "dataset": run_info["dataset"],
            "epoch": int(epoch_val),
        }
        for col in classified["val_metrics"]:
            row[col] = _parse_float(raw.get(col))
        for col in classified["train_losses"]:
            row[col] = _parse_float(raw.get(col))
        for col in classified["bpgs_weights"]:
            row[col] = _parse_float(raw.get(col))
        for col in classified["gradient_health"]:
            row[col] = _parse_float(raw.get(col))
        for col in classified["lr"]:
            row[col] = _parse_float(raw.get(col))
        epoch_rows.append(row)
    return epoch_rows


def _extract_weight_trajectory(
    metrics_path: Path,
    run_info: Dict[str, Any],
    study_name: str,
) -> List[Dict[str, object]]:
    """Extract BPGS weight evolution (per-epoch per-task weights)."""
    raw_rows = _read_all_csv_rows(metrics_path)
    if not raw_rows:
        return []

    columns = list(raw_rows[0].keys())
    weight_cols = [c for c in columns if c.startswith("bpgs/weight_")]
    s_cols = [c for c in columns if c.startswith("bpgs/s_")]
    if not weight_cols and not s_cols:
        return []

    trajectory: List[Dict[str, object]] = []
    for raw in raw_rows:
        epoch_val = _parse_float(raw.get("epoch"))
        if epoch_val is None:
            continue
        row: Dict[str, object] = {
            "study": study_name,
            "variant": run_info["variant"],
            "seed": run_info["training_seed"],
            "method": run_info["method"],
            "epoch": int(epoch_val),
        }
        for col in weight_cols:
            task_idx = col.replace("bpgs/weight_", "")
            row[f"weight_task_{task_idx}"] = _parse_float(raw.get(col))
        for col in s_cols:
            task_idx = col.replace("bpgs/s_", "")
            row[f"log_var_task_{task_idx}"] = _parse_float(raw.get(col))
        for agg_col in ("bpgs/weights_mean", "bpgs/weights_min", "bpgs/weights_max", "bpgs/total_loss"):
            if agg_col in raw:
                row[agg_col.replace("bpgs/", "bpgs_")] = _parse_float(raw.get(agg_col))
        trajectory.append(row)
    return trajectory


def _extract_gradient_diagnostics(
    metrics_path: Path,
    run_info: Dict[str, Any],
    study_name: str,
) -> List[Dict[str, object]]:
    """Extract gradient health monitoring data."""
    raw_rows = _read_all_csv_rows(metrics_path)
    if not raw_rows:
        return []

    columns = list(raw_rows[0].keys())
    health_cols = [c for c in columns if c.startswith("health/")]
    if not health_cols:
        return []

    diagnostics: List[Dict[str, object]] = []
    for raw in raw_rows:
        epoch_val = _parse_float(raw.get("epoch"))
        if epoch_val is None:
            continue
        row: Dict[str, object] = {
            "study": study_name,
            "variant": run_info["variant"],
            "seed": run_info["training_seed"],
            "method": run_info["method"],
            "epoch": int(epoch_val),
        }
        for col in health_cols:
            clean_name = col.replace("health/", "")
            row[clean_name] = _parse_float(raw.get(col))
        diagnostics.append(row)
    return diagnostics


# ---------------------------------------------------------------------------
# Per-task loss extraction
# ---------------------------------------------------------------------------

def _extract_per_task_losses(
    metrics_path: Path,
    run_info: Dict[str, Any],
    study_name: str,
) -> List[Dict[str, object]]:
    """Extract per-task training loss curves."""
    raw_rows = _read_all_csv_rows(metrics_path)
    if not raw_rows:
        return []

    columns = list(raw_rows[0].keys())
    task_loss_cols = [c for c in columns if c.startswith("train/") and c.endswith("_loss")]
    if not task_loss_cols:
        return []

    rows: List[Dict[str, object]] = []
    for raw in raw_rows:
        epoch_val = _parse_float(raw.get("epoch"))
        if epoch_val is None:
            continue
        for col in task_loss_cols:
            task_name = col.replace("train/", "").replace("_loss", "").replace("_norm", "_normalized")
            val = _parse_float(raw.get(col))
            if val is not None:
                rows.append({
                    "study": study_name,
                    "variant": run_info["variant"],
                    "seed": run_info["training_seed"],
                    "method": run_info["method"],
                    "epoch": int(epoch_val),
                    "task": task_name,
                    "loss": val,
                    "column": col,
                })
    return rows


# ---------------------------------------------------------------------------
# Selected-epoch records (backward-compatible, enhanced)
# ---------------------------------------------------------------------------

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


def _metrics_rows(metrics_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            val_total_loss = _parse_float(row.get("val/total_loss"))
            epoch_value = _parse_float(row.get("epoch"))
            if val_total_loss is None or epoch_value is None:
                continue
            rows.append({
                "epoch": int(epoch_value),
                "val/total_loss": val_total_loss,
                "val/miou": _parse_float(row.get("val/miou")),
                "val/depth_abs_rel": _parse_float(row.get("val/depth_abs_rel")),
                "val/normals_mean_angle": _parse_float(row.get("val/normals_mean_angle")),
            })
    return rows


def _build_selected_records(
    study_name: str,
    study_root: Path,
) -> List[Dict[str, object]]:
    """Build per-run records at the selected checkpoint (backward-compatible + enhanced)."""
    records: List[Dict[str, object]] = []
    if not study_root.exists():
        return records

    for run_dir in study_root.glob("*/*"):
        metadata_path = run_dir / "metadata.json"
        metrics_candidates = list((run_dir / "csv_logs").glob("*/*.csv"))
        if not metadata_path.exists() or not metrics_candidates:
            continue

        run_info = _extract_run_info(run_dir)
        if run_info is None:
            continue

        run_summary = _read_json(run_dir / "run_summary.json") if (run_dir / "run_summary.json").exists() else {}
        rows = _metrics_rows(metrics_candidates[0])
        if not rows:
            continue

        row = _select_row(rows, _selected_epoch(run_summary))
        run_info["study"] = study_name

        record: Dict[str, object] = {
            **run_info,
            "selected_epoch": row["epoch"],
            "miou": row.get("val/miou"),
            "abs_rel": row.get("val/depth_abs_rel"),
            "mean_angle_deg": row.get("val/normals_mean_angle"),
            "val_total_loss": row.get("val/total_loss"),
        }

        # Discover additional val metrics beyond the standard 4
        all_cols = _discover_csv_columns(metrics_candidates[0])
        extra_val = [c for c in all_cols if c.startswith("val/") and c not in (
            "val/total_loss", "val/miou", "val/depth_abs_rel", "val/normals_mean_angle"
        )]
        if extra_val:
            raw_rows = _read_all_csv_rows(metrics_candidates[0])
            selected_raw = None
            for raw in raw_rows:
                if _parse_float(raw.get("epoch")) == row["epoch"]:
                    selected_raw = raw
                    break
            if selected_raw:
                for col in extra_val:
                    val = _parse_float(selected_raw.get(col))
                    if val is not None:
                        record[col] = val

        records.append(record)

    # Compute delta_m against static baseline
    baselines = {
        int(r["training_seed"]): r
        for r in records
        if r["variant"] == "static" and r["training_seed"] is not None
    }
    for record in records:
        seed = record.get("training_seed")
        baseline = baselines.get(int(seed)) if seed is not None else None
        if baseline and all(record.get(k) is not None for k in ("miou", "abs_rel", "mean_angle_deg")):
            record["delta_m"] = compute_delta_m(
                results={
                    "miou": float(record["miou"]),
                    "abs_rel": float(record["abs_rel"]),
                    "mean_angle_deg": float(record["mean_angle_deg"]),
                },
                baseline={
                    "miou": float(baseline["miou"]),
                    "abs_rel": float(baseline["abs_rel"]),
                    "mean_angle_deg": float(baseline["mean_angle_deg"]),
                },
                metric_configs={"miou": False, "abs_rel": True, "mean_angle_deg": True},
            )
        else:
            record["delta_m"] = float("nan")

    return records


# ---------------------------------------------------------------------------
# Summary aggregation (enhanced with CI and more metrics)
# ---------------------------------------------------------------------------

TRAINING_SUMMARY_METRICS = [
    "miou", "abs_rel", "mean_angle_deg", "delta_m",
    "runtime_seconds", "stopped_epoch", "val_total_loss",
]


def _build_summary(
    records: List[Dict[str, object]],
    study_name: str,
    confidence: float = 0.95,
) -> List[Dict[str, object]]:
    """Build variant-level summary with mean, std, and confidence intervals."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["variant"])].append(record)

    # Discover all numeric metrics across records
    all_metrics: Set[str] = set()
    for record in records:
        for key, val in record.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if not math.isnan(float(val)) if isinstance(val, float) else True:
                    all_metrics.add(key)
    # Focus on meaningful metrics
    metric_fields = [m for m in TRAINING_SUMMARY_METRICS if m in all_metrics]
    # Add any extra val/ metrics found
    extra_val_metrics = [m for m in sorted(all_metrics) if m.startswith("val/") and m not in metric_fields]
    metric_fields.extend(extra_val_metrics)

    summary: List[Dict[str, object]] = []
    for variant, variant_records in sorted(grouped.items()):
        row: Dict[str, object] = {
            "study": study_name,
            "variant": variant,
            "method": variant_records[0].get("method"),
            "dataset": variant_records[0].get("dataset"),
            "n_runs": len(variant_records),
        }
        for metric in metric_fields:
            values = [
                float(r[metric])
                for r in variant_records
                if r.get(metric) is not None and not (
                    isinstance(r[metric], float) and math.isnan(float(r[metric]))
                )
            ]
            row[f"{metric}_mean"] = _metric_mean(values) if values else float("nan")
            row[f"{metric}_std"] = _metric_std(values) if values else float("nan")
            if len(values) >= 2:
                _, ci_lo, ci_hi = confidence_interval(values, confidence)
                row[f"{metric}_ci_lower"] = ci_lo
                row[f"{metric}_ci_upper"] = ci_hi
            else:
                row[f"{metric}_ci_lower"] = float("nan")
                row[f"{metric}_ci_upper"] = float("nan")
        summary.append(row)
    return summary


# ---------------------------------------------------------------------------
# Full training aggregation
# ---------------------------------------------------------------------------

def aggregate_training_full(
    study_name: str,
    confidence: float = 0.95,
    baseline_variant: str = "static",
) -> Dict[str, object]:
    """
    Comprehensive training study aggregation.

    Returns:
        records: per-run records at selected checkpoint (enhanced)
        summary: variant-level mean/std/CI
        epoch_curves: full epoch-level data for all runs
        weight_trajectories: BPGS weight evolution
        gradient_diagnostics: gradient health data
        per_task_losses: per-task training loss curves
        statistical_tests: pairwise significance tests vs baseline
        confidence_intervals: CI for each (variant, metric)
    """
    study_root = resolve_study_output_root(study_name)
    if not study_root.exists():
        return {
            "records": [], "summary": [], "epoch_curves": [],
            "weight_trajectories": [], "gradient_diagnostics": [],
            "per_task_losses": [], "statistical_tests": [],
            "confidence_intervals": [],
        }

    # 1. Selected-epoch records (backward-compatible + enhanced)
    records = _build_selected_records(study_name, study_root)

    # 2. Variant-level summary with CI
    summary = _build_summary(records, study_name, confidence)

    # 3. Full epoch-level curves
    all_epoch_curves: List[Dict[str, object]] = []
    all_weight_trajectories: List[Dict[str, object]] = []
    all_gradient_diagnostics: List[Dict[str, object]] = []
    all_per_task_losses: List[Dict[str, object]] = []

    for run_dir in study_root.glob("*/*"):
        metadata_path = run_dir / "metadata.json"
        metrics_candidates = list((run_dir / "csv_logs").glob("*/*.csv"))
        if not metadata_path.exists() or not metrics_candidates:
            continue

        run_info = _extract_run_info(run_dir)
        if run_info is None:
            continue

        metrics_path = metrics_candidates[0]
        all_epoch_curves.extend(_extract_epoch_curves(metrics_path, run_info, study_name))
        all_weight_trajectories.extend(_extract_weight_trajectory(metrics_path, run_info, study_name))
        all_gradient_diagnostics.extend(_extract_gradient_diagnostics(metrics_path, run_info, study_name))
        all_per_task_losses.extend(_extract_per_task_losses(metrics_path, run_info, study_name))

    # 4. Statistical tests
    stat_metrics = [m for m in TRAINING_SUMMARY_METRICS if any(r.get(m) is not None for r in records)]
    stat_tests = pairwise_tests(records, baseline_variant, stat_metrics, confidence=confidence)

    # 5. Confidence intervals per (variant, metric)
    ci_rows = compute_confidence_intervals(records, stat_metrics, confidence=confidence)

    return {
        "records": records,
        "summary": summary,
        "epoch_curves": all_epoch_curves,
        "weight_trajectories": all_weight_trajectories,
        "gradient_diagnostics": all_gradient_diagnostics,
        "per_task_losses": all_per_task_losses,
        "statistical_tests": stat_tests,
        "confidence_intervals": ci_rows,
    }


# ---------------------------------------------------------------------------
# Empirical aggregation (enhanced)
# ---------------------------------------------------------------------------

EMPIRICAL_NUMERIC_FIELDS = [
    "macro_score",
    "worst_task_score",
    "best_task_score",
    "score_std",
    "train_loss_mean",
    "train_loss_variance",
    "val_loss_mean",
    "val_loss_variance",
    "grad_norm_mean",
    "grad_cosine_mean",
    "train_total_variation",
    "val_total_variation",
    "weight_entropy_mean",
    "weight_entropy_min",
    "weight_min",
    "weight_max",
    "latent_span",
]


def aggregate_empirical_full(
    study_name: str,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """Comprehensive empirical study aggregation."""
    study_root = resolve_study_output_root(study_name)
    records: List[Dict[str, object]] = []

    if not study_root.exists():
        return {
            "records": [], "summary": [],
            "statistical_tests": [], "confidence_intervals": [],
            "history_data": [],
        }

    for results_path in study_root.glob("*/*/*/metrics/results.csv"):
        variant = results_path.parents[3].name
        seed_match = re.search(r"seed_(\d+)$", results_path.parents[1].name)
        seed = int(seed_match.group(1)) if seed_match else None
        with results_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["study"] = study_name
                row["variant"] = variant
                row["empirical_seed"] = seed
                records.append(dict(row))

    # Summary with CI
    grouped: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    for record in records:
        grouped[(str(record["variant"]), str(record["method"]))].append(record)

    summary_rows: List[Dict[str, object]] = []
    for (variant, method), group in sorted(grouped.items()):
        row: Dict[str, object] = {
            "study": study_name,
            "variant": variant,
            "method": method,
            "n_runs": len(group),
        }
        for field in EMPIRICAL_NUMERIC_FIELDS:
            values = [_parse_float(str(item.get(field, ""))) for item in group]
            clean = [v for v in values if v is not None]
            row[f"{field}_mean"] = _metric_mean(clean) if clean else float("nan")
            row[f"{field}_std"] = _metric_std(clean) if clean else float("nan")
            if len(clean) >= 2:
                _, ci_lo, ci_hi = confidence_interval(clean, confidence)
                row[f"{field}_ci_lower"] = ci_lo
                row[f"{field}_ci_upper"] = ci_hi
            else:
                row[f"{field}_ci_lower"] = float("nan")
                row[f"{field}_ci_upper"] = float("nan")
        summary_rows.append(row)

    # Statistical tests - compare each method against first method per variant
    stat_tests: List[Dict[str, object]] = []
    for (variant, _), group in sorted(grouped.items()):
        methods_in_variant = sorted(set(str(r["method"]) for r in group))
        if len(methods_in_variant) < 2:
            continue
        baseline_method = methods_in_variant[0]
        for metric in EMPIRICAL_NUMERIC_FIELDS:
            baseline_vals = [
                float(r[metric]) for r in group
                if str(r["method"]) == baseline_method
                and _parse_float(str(r.get(metric, ""))) is not None
            ]
            if len(baseline_vals) < 2:
                continue
            for method in methods_in_variant[1:]:
                method_vals = [
                    float(r[metric]) for r in group
                    if str(r["method"]) == method
                    and _parse_float(str(r.get(metric, ""))) is not None
                ]
                if len(method_vals) < 2:
                    continue
                from analysis.bpgs_study.statistics import welch_t_test, cohens_d
                t_result = welch_t_test(method_vals, baseline_vals)
                d = cohens_d(method_vals, baseline_vals)
                stat_tests.append({
                    "variant": variant,
                    "metric": metric,
                    "method": method,
                    "baseline_method": baseline_method,
                    "delta_mean": float(np.mean(method_vals)) - float(np.mean(baseline_vals)),
                    "cohens_d": d,
                    **t_result,
                })

    # CI per (variant, method, metric)
    ci_rows: List[Dict[str, object]] = []
    for (variant, method), group in sorted(grouped.items()):
        for field in EMPIRICAL_NUMERIC_FIELDS:
            values = [_parse_float(str(item.get(field, ""))) for item in group]
            clean = [v for v in values if v is not None]
            if not clean:
                continue
            mean, ci_lo, ci_hi = confidence_interval(clean, confidence)
            ci_rows.append({
                "variant": variant,
                "method": method,
                "metric": field,
                "n": len(clean),
                "mean": mean,
                "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            })

    # Collect history data for figure generation
    history_data: List[Dict[str, object]] = []
    for results_path in study_root.glob("*/*/*/artifacts/result_manifest.json"):
        variant = results_path.parents[3].name
        try:
            manifest = _read_json(results_path)
            if isinstance(manifest, list):
                for entry in manifest:
                    entry["variant"] = variant
                    history_data.append(entry)
        except Exception:
            pass

    return {
        "records": records,
        "summary": summary_rows,
        "statistical_tests": stat_tests,
        "confidence_intervals": ci_rows,
        "history_data": history_data,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def aggregate_study_full(
    study: StudySpec,
    confidence: float = 0.95,
    baseline_variant: str = "static",
) -> Dict[str, object]:
    """Full aggregation for any study type."""
    if study.kind == "training":
        return aggregate_training_full(study.name, confidence, baseline_variant)
    if study.kind == "empirical":
        return aggregate_empirical_full(study.name, confidence)
    raise ValueError(f"Unsupported study kind: {study.kind}")
