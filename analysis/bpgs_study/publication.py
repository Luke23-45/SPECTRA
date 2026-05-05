from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

_SIG_MARKS = {0.001: "***", 0.01: "**", 0.05: "*", 0.1: "\\dagger"}


def _significance_mark(p_value: float) -> str:
    for threshold, mark in _SIG_MARKS.items():
        if p_value < threshold:
            return mark
    return ""


def _fmt_mean_std(mean: float, std: float, sig_mark: str = "", bold: bool = False) -> str:
    """Format mean±std with optional significance mark and bold."""
    if math.isnan(mean):
        return "---"
    text = f"{mean:.4f} \\pm {std:.4f}"
    if sig_mark:
        text += sig_mark
    if bold:
        text = f"\\textbf{{{text}}}"
    return text


def _find_best_variant(
    summary: List[Dict[str, object]],
    metric: str,
    higher_is_better: bool = True,
) -> str:
    """Find the variant with the best mean for a metric."""
    best_variant = ""
    best_val = float("-inf") if higher_is_better else float("inf")
    for row in summary:
        mean_key = f"{metric}_mean"
        if mean_key not in row:
            continue
        val = float(row[mean_key])
        if math.isnan(val):
            continue
        if higher_is_better and val > best_val:
            best_val = val
            best_variant = str(row["variant"])
        elif not higher_is_better and val < best_val:
            best_val = val
            best_variant = str(row["variant"])
    return best_variant


def generate_latex_main_table(
    summary: List[Dict[str, object]],
    statistical_tests: List[Dict[str, object]],
    metrics: Sequence[str],
    metric_labels: Optional[Dict[str, str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    caption: str = "Main results.",
    label: str = "tab:main_results",
    baseline_variant: str = "static",
) -> str:
    """
    Generate a LaTeX table with mean±std, significance marks, and bold best.

    Columns: Method | metric_1 | metric_2 | ...
    """
    if metric_labels is None:
        metric_labels = {m: m for m in metrics}
    if higher_is_better is None:
        higher_is_better = {m: True for m in metrics}

    # Build significance lookup: (variant, metric) -> corrected p-value
    sig_lookup: Dict[tuple[str, str], float] = {}
    for test in statistical_tests:
        variant = str(test.get("variant", ""))
        metric = str(test.get("metric", ""))
        p_corr = test.get("p_value_corrected")
        if p_corr is not None:
            sig_lookup[(variant, metric)] = float(p_corr)

    # Find best variant per metric
    best_per_metric: Dict[str, str] = {}
    for metric in metrics:
        best_per_metric[metric] = _find_best_variant(
            summary, metric, higher_is_better.get(metric, True)
        )

    # Column format
    n_cols = 1 + len(metrics)
    col_fmt = "l" + "c" * len(metrics)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append("  \\toprule")

    # Header
    header = "Method"
    for metric in metrics:
        header += f" & {metric_labels.get(metric, metric)}"
    header += " \\\\"
    lines.append(f"  {header}")
    lines.append("  \\midrule")

    # Data rows
    for row in summary:
        variant = str(row.get("variant", ""))
        method = str(row.get("method", variant))
        line = method
        for metric in metrics:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            mean = float(row.get(mean_key, float("nan")))
            std = float(row.get(std_key, float("nan")))
            sig = _significance_mark(sig_lookup.get((variant, metric), 1.0))
            is_best = variant == best_per_metric.get(metric, "")
            line += f" & {_fmt_mean_std(mean, std, sig, bold=is_best)}"
        line += " \\\\"
        lines.append(f"  {line}")

    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")

    # Significance note
    notes = []
    for threshold, mark in _SIG_MARKS.items():
        notes.append(f"{mark}: $p < {threshold}$ (corrected)")
    lines.append(f"  \\par\\vspace{{2pt}}{{\\small {', '.join(notes)}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_latex_ablation_table(
    summary: List[Dict[str, object]],
    statistical_tests: List[Dict[str, object]],
    metrics: Sequence[str],
    metric_labels: Optional[Dict[str, str]] = None,
    caption: str = "Ablation study results.",
    label: str = "tab:ablation",
    baseline_variant: str = "bpgs_canonical",
) -> str:
    """Generate ablation table with delta from canonical variant."""
    if metric_labels is None:
        metric_labels = {m: m for m in metrics}

    # Find canonical means
    canonical_means: Dict[str, float] = {}
    for row in summary:
        if str(row.get("variant")) == baseline_variant:
            for metric in metrics:
                canonical_means[metric] = float(row.get(f"{metric}_mean", float("nan")))
            break

    sig_lookup: Dict[tuple[str, str], float] = {}
    for test in statistical_tests:
        variant = str(test.get("variant", ""))
        metric = str(test.get("metric", ""))
        p_corr = test.get("p_value_corrected")
        if p_corr is not None:
            sig_lookup[(variant, metric)] = float(p_corr)

    col_fmt = "l" + "c" * len(metrics)
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append("  \\toprule")
    header = "Configuration"
    for metric in metrics:
        header += f" & {metric_labels.get(metric, metric)}"
    header += " \\\\"
    lines.append(f"  {header}")
    lines.append("  \\midrule")

    for row in summary:
        variant = str(row.get("variant", ""))
        escaped_variant = variant.replace("_", "\\_")
        line = escaped_variant
        for metric in metrics:
            mean = float(row.get(f"{metric}_mean", float("nan")))
            std = float(row.get(f"{metric}_std", float("nan")))
            delta = mean - canonical_means.get(metric, float("nan"))
            sig = _significance_mark(sig_lookup.get((variant, metric), 1.0))
            if math.isnan(mean):
                line += " & ---"
            else:
                delta_str = f"({delta:+.4f})" if not math.isnan(delta) else ""
                line += f" & {mean:.4f}$\\pm${std:.4f} {delta_str}{sig}"
        line += " \\\\"
        lines.append(f"  {line}")

    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    escaped_baseline = baseline_variant.replace("_", "\\_")
    lines.append(f"  \\par\\vspace{{2pt}}{{\\small Deltas relative to {escaped_baseline}. "
                 "{{$^{**}$}: $p<0.01$, {$^*$}: $p<0.05$ (corrected).}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_latex_scale_table(
    summary: List[Dict[str, object]],
    metrics: Sequence[str],
    scale_variants: Optional[List[str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    caption: str = "Scale robustness results across loss rescaling factors.",
    label: str = "tab:scale_robustness",
) -> str:
    """Generate scale robustness table showing degradation across scale factors."""
    if metric_labels is None:
        metric_labels = {m: m for m in metrics}

    col_fmt = "ll" + "c" * len(metrics)
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append("  \\toprule")
    header = "Scale & Method"
    for metric in metrics:
        header += f" & {metric_labels.get(metric, metric)}"
    header += " \\\\"
    lines.append(f"  {header}")
    lines.append("  \\midrule")

    # Group by scale variant, then method
    grouped: Dict[str, List[Dict]] = {}
    for row in summary:
        variant = str(row.get("variant", ""))
        grouped.setdefault(variant, []).append(row)

    if scale_variants is None:
        scale_variants = sorted(grouped.keys())

    for scale in scale_variants:
        if scale not in grouped:
            continue
        for i, row in enumerate(grouped[scale]):
            method = str(row.get("method", ""))
            scale_label = scale if i == 0 else ""
            line = f"{scale_label} & {method}"
            for metric in metrics:
                mean = float(row.get(f"{metric}_mean", float("nan")))
                std = float(row.get(f"{metric}_std", float("nan")))
                if math.isnan(mean):
                    line += " & ---"
                else:
                    line += f" & {mean:.4f}$\\pm${std:.4f}"
            line += " \\\\"
            lines.append(f"  {line}")
        if scale != scale_variants[-1]:
            lines.append("  \\midrule")

    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure data generation (CSV files for plotting)
# ---------------------------------------------------------------------------

def generate_learning_curve_data(
    epoch_curves: List[Dict[str, object]],
    metrics: Sequence[str],
    output_path: Path,
) -> None:
    """
    Write CSV data for learning curve plots.

    Columns: study, variant, method, seed, epoch, <metric_1>, <metric_2>, ...
    """
    if not epoch_curves:
        return

    # Filter to requested metrics + identity columns
    identity_cols = ["study", "variant", "method", "seed", "epoch"]
    available_metrics = set()
    for row in epoch_curves:
        for k, v in row.items():
            if k not in identity_cols and v is not None:
                available_metrics.add(k)

    selected_metrics = [m for m in metrics if m in available_metrics]
    if not selected_metrics:
        selected_metrics = sorted(available_metrics)

    cols = identity_cols + selected_metrics
    filtered_rows = []
    for row in epoch_curves:
        filtered = {c: row.get(c) for c in cols}
        filtered_rows.append(filtered)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(filtered_rows)


def generate_weight_evolution_data(
    weight_trajectories: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """Write CSV data for weight evolution plots."""
    if not weight_trajectories:
        return
    cols = list(weight_trajectories[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(weight_trajectories)


def generate_radar_chart_data(
    summary: List[Dict[str, object]],
    metrics: Sequence[str],
    output_path: Path,
) -> None:
    """
    Write CSV data for radar/spider charts.

    Columns: variant, method, metric, mean, std
    """
    rows: List[Dict[str, object]] = []
    for row in summary:
        variant = str(row.get("variant", ""))
        method = str(row.get("method", variant))
        for metric in metrics:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in row:
                rows.append({
                    "variant": variant,
                    "method": method,
                    "metric": metric,
                    "mean": row[mean_key],
                    "std": row.get(std_key, 0.0),
                })
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "method", "metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(rows)


def generate_convergence_data(
    epoch_curves: List[Dict[str, object]],
    target_metric: str,
    target_fractions: Sequence[float] = (0.90, 0.95, 0.99),
    output_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """
    Compute convergence speed: epoch at which each run reaches
    target_fraction × final_value of target_metric.

    Returns rows: study, variant, method, seed, fraction, epoch_reached, final_value
    """
    # Group curves by (study, variant, seed)
    grouped: Dict[tuple, List[Dict]] = {}
    for row in epoch_curves:
        key = (row.get("study"), row.get("variant"), row.get("method"), row.get("seed"))
        grouped.setdefault(key, []).append(row)

    results: List[Dict[str, object]] = []
    for (study, variant, method, seed), curves in grouped.items():
        curves_sorted = sorted(curves, key=lambda r: int(r.get("epoch", 0)))
        # Get final value
        final_val = None
        for curve_row in reversed(curves_sorted):
            val = curve_row.get(target_metric)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                final_val = float(val)
                break
        if final_val is None:
            continue

        for frac in target_fractions:
            target_val = final_val * frac
            epoch_reached = None
            for curve_row in curves_sorted:
                val = curve_row.get(target_metric)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    # For metrics where higher is better
                    if float(val) >= target_val:
                        epoch_reached = int(curve_row.get("epoch", 0))
                        break
            results.append({
                "study": study,
                "variant": variant,
                "method": method,
                "seed": seed,
                "metric": target_metric,
                "fraction": frac,
                "epoch_reached": epoch_reached,
                "final_value": final_val,
            })

    if output_path is not None and results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return results


def generate_scale_degradation_data(
    summary: List[Dict[str, object]],
    metrics: Sequence[str],
    baseline_variant: str = "x1",
    output_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """
    Compute degradation relative to baseline scale variant.

    Returns rows: variant, method, metric, mean, relative_to_baseline
    """
    # Find baseline means per (method, metric)
    baseline_means: Dict[tuple[str, str], float] = {}
    for row in summary:
        if str(row.get("variant")) == baseline_variant:
            method = str(row.get("method", ""))
            for metric in metrics:
                mean_key = f"{metric}_mean"
                if mean_key in row:
                    baseline_means[(method, metric)] = float(row[mean_key])

    results: List[Dict[str, object]] = []
    for row in summary:
        variant = str(row.get("variant", ""))
        method = str(row.get("method", ""))
        for metric in metrics:
            mean_key = f"{metric}_mean"
            if mean_key not in row:
                continue
            mean = float(row[mean_key])
            base = baseline_means.get((method, metric))
            relative = mean / base if base and base != 0 and not math.isnan(base) else None
            results.append({
                "variant": variant,
                "method": method,
                "metric": metric,
                "mean": mean,
                "baseline_mean": base,
                "relative_to_baseline": relative,
            })

    if output_path is not None and results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return results


# ---------------------------------------------------------------------------
# Write all publication artifacts for a study
# ---------------------------------------------------------------------------

def write_publication_artifacts(
    study_name: str,
    study_kind: str,
    aggregated: Dict[str, object],
    report_dir: Path,
) -> None:
    """Write all publication artifacts for one study to report_dir."""
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregated.get("summary", [])
    stat_tests = aggregated.get("statistical_tests", [])
    epoch_curves = aggregated.get("epoch_curves", [])

    if study_kind == "training":
        # Discover available metrics from summary
        available_metrics = set()
        for row in summary:
            for key in row:
                if key.endswith("_mean") and key != "n_runs":
                    metric = key.replace("_mean", "")
                    available_metrics.add(metric)

        core_metrics = [m for m in ["miou", "abs_rel", "mean_angle_deg", "delta_m"] if m in available_metrics]
        metric_labels = {
            "miou": "mIoU$\\uparrow$",
            "abs_rel": "Abs Rel$\\downarrow$",
            "mean_angle_deg": "Mean Angle$\\downarrow$",
            "delta_m": "$\\Delta_m$\\%$\\uparrow$",
            "val_total_loss": "Val Loss$\\downarrow$",
            "runtime_seconds": "Time (s)",
            "stopped_epoch": "Epochs",
        }
        higher_is_better = {
            "miou": True, "delta_m": True,
            "abs_rel": False, "mean_angle_deg": False,
            "val_total_loss": False, "runtime_seconds": False,
            "stopped_epoch": False,
        }

        # Main results table
        if core_metrics:
            tex = generate_latex_main_table(
                summary, stat_tests, core_metrics,
                metric_labels=metric_labels,
                higher_is_better=higher_is_better,
                caption=f"Results for {study_name}.",
                label=f"tab:{study_name}",
            )
            (tables_dir / "main_results.tex").write_text(tex, encoding="utf-8")

        # Learning curve data
        val_metrics = [m for m in available_metrics if m.startswith("val/")]
        if val_metrics or core_metrics:
            curve_metrics = val_metrics if val_metrics else ["val/total_loss"]
            generate_learning_curve_data(epoch_curves, curve_metrics, figures_dir / "learning_curves.csv")

        # Weight evolution data
        weight_traj = aggregated.get("weight_trajectories", [])
        if weight_traj:
            generate_weight_evolution_data(weight_traj, figures_dir / "weight_evolution.csv")

        # Radar chart data
        if core_metrics:
            generate_radar_chart_data(summary, core_metrics, figures_dir / "radar_data.csv")

        # Convergence data
        if epoch_curves and "val/total_loss" in {c for row in epoch_curves for c in row if row.get(c) is not None}:
            generate_convergence_data(
                epoch_curves, "val/total_loss",
                target_fractions=(0.90, 0.95),
                output_path=figures_dir / "convergence.csv",
            )

        # Per-task loss data
        per_task = aggregated.get("per_task_losses", [])
        if per_task:
            cols = list(per_task[0].keys())
            with (figures_dir / "per_task_losses.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=cols)
                writer.writeheader()
                writer.writerows(per_task)

        # Gradient diagnostics data
        grad_diag = aggregated.get("gradient_diagnostics", [])
        if grad_diag:
            cols = list(grad_diag[0].keys())
            with (figures_dir / "gradient_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=cols)
                writer.writeheader()
                writer.writerows(grad_diag)

    elif study_kind == "empirical":
        # Scale robustness / stress tables
        empirical_metrics = ["macro_score", "worst_task_score", "weight_entropy_mean", "weight_min", "weight_max"]
        available_emp_metrics = [m for m in empirical_metrics if any(
            f"{m}_mean" in row for row in summary
        )]
        if available_emp_metrics:
            metric_labels_emp = {
                "macro_score": "Macro Score$\\uparrow$",
                "worst_task_score": "Worst Task$\\uparrow$",
                "weight_entropy_mean": "Weight Entropy",
                "weight_min": "Min Weight",
                "weight_max": "Max Weight",
            }
            tex = generate_latex_scale_table(
                summary, available_emp_metrics,
                metric_labels=metric_labels_emp,
                caption=f"Stress test results for {study_name}.",
                label=f"tab:{study_name}",
            )
            (tables_dir / "scale_results.tex").write_text(tex, encoding="utf-8")

            # Scale degradation data
            generate_scale_degradation_data(
                summary, available_emp_metrics,
                baseline_variant="x1",
                output_path=figures_dir / "scale_degradation.csv",
            )

        # Radar data for empirical
        if available_emp_metrics:
            generate_radar_chart_data(summary, available_emp_metrics, figures_dir / "radar_data.csv")
