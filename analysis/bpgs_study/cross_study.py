from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from studies.bpgs_study.common.paths import OUTPUT_ROOT
from studies.bpgs_study.common.specs import StudySpec
from analysis.bpgs_study.aggregation_v2 import (
    aggregate_study_full,
    write_csv,
)
from analysis.bpgs_study.statistics import (
    cohens_d,
    confidence_interval,
    holm_bonferroni,
    pairwise_tests,
    welch_t_test,
)


# ---------------------------------------------------------------------------
# Cross-study unified table
# ---------------------------------------------------------------------------

def build_unified_records(
    studies: List[StudySpec],
    confidence: float = 0.95,
    baseline_variant: str = "static",
) -> Dict[str, object]:
    """
    Aggregate all studies and produce a unified view across datasets.

    Returns:
        unified_records: flat list of per-run records from all studies
        unified_summary: method × dataset summary with CI
        method_rankings: method rankings per dataset
        latex_main_table: unified LaTeX table for the paper
    """
    all_records: List[Dict[str, object]] = []
    all_summaries: List[Dict[str, object]] = []
    study_results: Dict[str, Dict[str, object]] = {}

    for study in studies:
        aggregated = aggregate_study_full(study, confidence, baseline_variant)
        study_results[study.name] = aggregated

        # Normalize records with study metadata
        for rec in aggregated.get("records", []):
            rec["study_name"] = study.name
            rec["study_group"] = study.group
            rec["study_kind"] = study.kind
            all_records.append(rec)

        for row in aggregated.get("summary", []):
            row["study_name"] = study.name
            row["study_group"] = study.group
            all_summaries.append(row)

    # Build method × dataset pivot
    pivot = _build_method_dataset_pivot(all_summaries)

    # Method rankings
    rankings = _compute_method_rankings(all_summaries)

    # Unified LaTeX table
    latex = _generate_unified_latex_table(all_summaries)

    # Cross-dataset statistical comparison
    cross_stats = _cross_dataset_statistics(all_records)

    return {
        "unified_records": all_records,
        "unified_summary": all_summaries,
        "method_dataset_pivot": pivot,
        "method_rankings": rankings,
        "latex_main_table": latex,
        "cross_dataset_statistics": cross_stats,
        "study_results": study_results,
    }


def _build_method_dataset_pivot(
    summary_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Build method × dataset pivot table.

    Each row = one method, columns = datasets, values = delta_m_mean (or other metric).
    """
    # Discover methods and datasets
    methods = sorted(set(str(r.get("method", "")) for r in summary_rows))
    datasets = sorted(set(str(r.get("dataset", "")) for r in summary_rows))

    # Group by (method, dataset) — may have multiple variants
    grouped: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    for row in summary_rows:
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        grouped[(method, dataset)].append(row)

    # For each (method, dataset), pick the best variant (by delta_m_mean or miou_mean)
    best_per_combo: Dict[tuple[str, str], Dict] = {}
    for (method, dataset), rows in grouped.items():
        best = None
        best_score = float("-inf")
        for row in rows:
            score = float(row.get("delta_m_mean", float("nan")))
            if math.isnan(score):
                score = float(row.get("miou_mean", float("nan")))
            if math.isnan(score):
                score = float(row.get("macro_score_mean", float("nan")))
            if not math.isnan(score) and score > best_score:
                best_score = score
                best = row
        if best is not None:
            best_per_combo[(method, dataset)] = best

    pivot_rows: List[Dict[str, object]] = []
    for method in methods:
        row: Dict[str, object] = {"method": method}
        for dataset in datasets:
            combo = best_per_combo.get((method, dataset))
            if combo:
                # Add all available mean metrics
                for key, val in combo.items():
                    if key.endswith("_mean") and key not in ("n_runs",):
                        row[f"{dataset}/{key}"] = val
            else:
                row[f"{dataset}/present"] = False
        pivot_rows.append(row)

    return pivot_rows


def _compute_method_rankings(
    summary_rows: List[Dict[str, object]],
    metric: str = "delta_m",
) -> List[Dict[str, object]]:
    """
    Rank methods per dataset based on a metric.

    Returns rows: dataset, method, rank, mean_value, n_variants_above
    """
    datasets = sorted(set(str(r.get("dataset", "")) for r in summary_rows))
    results: List[Dict[str, object]] = []

    for dataset in datasets:
        dataset_rows = [r for r in summary_rows if str(r.get("dataset")) == dataset]
        # Sort by metric mean (descending for delta_m, miou; ascending for abs_rel)
        metric_key = f"{metric}_mean"
        ranked = sorted(
            dataset_rows,
            key=lambda r: float(r.get(metric_key, float("-inf"))),
            reverse=(metric != "abs_rel"),
        )
        for rank, row in enumerate(ranked, start=1):
            results.append({
                "dataset": dataset,
                "method": str(row.get("method", "")),
                "variant": str(row.get("variant", "")),
                "rank": rank,
                "value": row.get(metric_key),
                "n_runs": row.get("n_runs"),
            })

    return results


def _generate_unified_latex_table(
    summary_rows: List[Dict[str, object]],
) -> str:
    """Generate the main paper table: method × dataset with delta_m."""
    methods = sorted(set(str(r.get("method", "")) for r in summary_rows))
    datasets = sorted(set(str(r.get("dataset", "")) for r in summary_rows))

    # Find best per (method, dataset)
    best_per_combo: Dict[tuple[str, str], Dict] = {}
    grouped: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    for row in summary_rows:
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        grouped[(method, dataset)].append(row)

    for (method, dataset), rows in grouped.items():
        best = max(rows, key=lambda r: float(r.get("delta_m_mean", float("-inf"))))
        best_per_combo[(method, dataset)] = best

    # Find best method per dataset
    best_method_per_dataset: Dict[str, str] = {}
    for dataset in datasets:
        best_val = float("-inf")
        best_method = ""
        for method in methods:
            combo = best_per_combo.get((method, dataset))
            if combo:
                val = float(combo.get("delta_m_mean", float("-inf")))
                if not math.isnan(val) and val > best_val:
                    best_val = val
                    best_method = method
        best_method_per_dataset[dataset] = best_method

    col_fmt = "l" + "c" * len(datasets)
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Cross-dataset comparison of multi-task weighting methods.}")
    lines.append("  \\label{tab:cross_dataset}")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append("  \\toprule")
    header = "Method"
    for dataset in datasets:
        header += f" & {dataset.capitalize()}"
    header += " \\\\"
    lines.append(f"  {header}")
    lines.append("  \\midrule")

    for method in methods:
        line = method.capitalize()
        for dataset in datasets:
            combo = best_per_combo.get((method, dataset))
            if combo is None:
                line += " & ---"
                continue
            mean = float(combo.get("delta_m_mean", float("nan")))
            std = float(combo.get("delta_m_std", float("nan")))
            if math.isnan(mean):
                # Fall back to miou
                mean = float(combo.get("miou_mean", float("nan")))
                std = float(combo.get("miou_std", float("nan")))
            if math.isnan(mean):
                line += " & ---"
                continue
            is_best = method == best_method_per_dataset.get(dataset, "")
            text = f"{mean:.2f} $\\pm$ {std:.2f}"
            if is_best:
                text = f"\\textbf{{{text}}}"
            line += f" & {text}"
        line += " \\\\"
        lines.append(f"  {line}")

    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\par\\vspace{2pt}{\\small $\\Delta_m$\\% (higher is better). "
                 "Best in \\textbf{bold}.}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def _cross_dataset_statistics(
    all_records: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    For each pair of methods, test if one consistently outperforms the other
    across all datasets (paired by seed).
    """
    # Group records by (method, dataset, seed)
    records_by_key: Dict[tuple, List[Dict]] = defaultdict(list)
    for rec in all_records:
        method = str(rec.get("method", ""))
        dataset = str(rec.get("dataset", ""))
        seed = rec.get("training_seed") or rec.get("empirical_seed")
        if seed is None:
            continue
        records_by_key[(method, dataset, int(seed))].append(rec)

    # Get all methods
    methods = sorted(set(str(r.get("method", "")) for r in all_records))
    if len(methods) < 2:
        return []

    # For each metric, collect paired differences across datasets
    metrics_to_test = ["delta_m", "miou", "abs_rel"]
    available_metrics = [m for m in metrics_to_test if any(
        r.get(m) is not None and not (isinstance(r.get(m), float) and math.isnan(float(r.get(m))))
        for r in all_records
    )]

    results: List[Dict[str, object]] = []
    baseline_method = methods[0]

    for metric in available_metrics:
        for method in methods[1:]:
            # Collect paired values across all datasets
            baseline_vals = []
            method_vals = []
            datasets = sorted(set(str(r.get("dataset", "")) for r in all_records))

            for dataset in datasets:
                # Find matching seeds
                baseline_seeds = {
                    int(seed): rec
                    for (m, d, seed), recs in records_by_key.items()
                    if m == baseline_method and d == dataset
                    for rec in recs
                }
                method_seeds = {
                    int(seed): rec
                    for (m, d, seed), recs in records_by_key.items()
                    if m == method and d == dataset
                    for rec in recs
                }
                common_seeds = set(baseline_seeds.keys()) & set(method_seeds.keys())
                for seed in common_seeds:
                    b_val = baseline_seeds[seed].get(metric)
                    m_val = method_seeds[seed].get(metric)
                    if (b_val is not None and m_val is not None
                            and not math.isnan(float(b_val))
                            and not math.isnan(float(m_val))):
                        baseline_vals.append(float(b_val))
                        method_vals.append(float(m_val))

            if len(baseline_vals) < 2:
                continue

            t_result = welch_t_test(method_vals, baseline_vals)
            d = cohens_d(method_vals, baseline_vals)
            results.append({
                "metric": metric,
                "method": method,
                "baseline_method": baseline_method,
                "n_paired_samples": len(baseline_vals),
                "method_mean": float(np.mean(method_vals)),
                "baseline_mean": float(np.mean(baseline_vals)),
                "delta_mean": float(np.mean(method_vals)) - float(np.mean(baseline_vals)),
                "cohens_d": d,
                **t_result,
            })

    # Holm-Bonferroni correction
    if results:
        raw_ps = [r["p_value"] for r in results]
        corrected = holm_bonferroni(raw_ps)
        for row, cp in zip(results, corrected):
            row["p_value_corrected"] = cp
            row["significant_005"] = cp < 0.05

    return results


# ---------------------------------------------------------------------------
# Write cross-study report
# ---------------------------------------------------------------------------

def write_cross_study_report(
    studies: List[StudySpec],
    output_dir: Path,
    confidence: float = 0.95,
    baseline_variant: str = "static",
) -> Path:
    """Write the complete cross-study comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    unified = build_unified_records(studies, confidence, baseline_variant)

    # Write unified records
    records = unified["unified_records"]
    if records:
        write_csv(output_dir / "unified_records.csv", records)

    # Write unified summary
    summary = unified["unified_summary"]
    if summary:
        write_csv(output_dir / "unified_summary.csv", summary)

    # Write method × dataset pivot
    pivot = unified["method_dataset_pivot"]
    if pivot:
        write_csv(output_dir / "method_dataset_pivot.csv", pivot)

    # Write method rankings
    rankings = unified["method_rankings"]
    if rankings:
        write_csv(output_dir / "method_rankings.csv", rankings)

    # Write cross-dataset statistics
    cross_stats = unified["cross_dataset_statistics"]
    if cross_stats:
        write_csv(output_dir / "cross_dataset_statistics.csv", cross_stats)

    # Write LaTeX main table
    latex = unified["latex_main_table"]
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables" / "cross_dataset_main.tex").write_text(latex, encoding="utf-8")

    # Write per-study detailed reports
    study_results = unified["study_results"]
    for study_name, aggregated in study_results.items():
        study_dir = output_dir / study_name
        study_dir.mkdir(parents=True, exist_ok=True)

        study_spec = next((s for s in studies if s.name == study_name), None)
        study_kind = study_spec.kind if study_spec else "training"

        from analysis.bpgs_study.publication import write_publication_artifacts
        write_publication_artifacts(study_name, study_kind, aggregated, study_dir)

        # Write all CSV data
        for key in ("records", "summary", "epoch_curves", "weight_trajectories",
                     "gradient_diagnostics", "per_task_losses", "statistical_tests",
                     "confidence_intervals"):
            data = aggregated.get(key, [])
            if data:
                write_csv(study_dir / f"{key}.csv", data)

        # Write full JSON
        (study_dir / "full_aggregation.json").write_text(
            json.dumps(aggregated, indent=2, default=str),
            encoding="utf-8",
        )

    # Write manifest
    manifest = {
        "report_type": "cross_study_comparison",
        "studies": [s.name for s in studies],
        "confidence": confidence,
        "baseline_variant": baseline_variant,
        "output_dir": str(output_dir),
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return output_dir
