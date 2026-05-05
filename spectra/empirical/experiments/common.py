"""Shared execution helpers for config-driven empirical experiments."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from spectra.empirical.analysis import summarize_comparison_frame
from spectra.empirical.analysis.plotters import plot_loss_trajectory, plot_metric_boxplot, plot_metric_heatmap
from spectra.empirical.runners.method_comparison import MethodComparisonHarness
from spectra.empirical.utils import (
    EmpiricalRunContext,
    build_comparison_config,
    build_generator_config,
    validate_empirical_config,
    write_frame_csv,
    write_frame_json,
    write_json,
    write_jsonl,
)


def run_single_experiment(cfg: DictConfig, run_context: EmpiricalRunContext, logger: logging.Logger) -> Dict[str, object]:
    validate_empirical_config(cfg)
    generator_config = build_generator_config(cfg)
    comparison_config = build_comparison_config(cfg, output_dir=str(run_context.artifacts_dir))
    harness = MethodComparisonHarness(generator_config, comparison_config)
    logger.info(
        "Running experiment '%s' with methods=%s seeds=%s",
        cfg.experiment.name,
        list(cfg.methods["names"]),
        list(cfg.seeds["values"]),
    )
    results = harness.run()
    frame = harness.to_frame(results)
    summary = summarize_comparison_frame(frame, metric=str(cfg.analysis.primary_metric))

    write_frame_csv(run_context.metrics_dir / "results.csv", frame)
    write_frame_json(run_context.metrics_dir / "results.json", frame)
    write_jsonl(run_context.metrics_dir / "results.jsonl", frame.to_dict(orient="records"))
    write_frame_csv(run_context.metrics_dir / "summary.csv", summary)
    write_frame_json(run_context.metrics_dir / "summary.json", summary)
    write_json(run_context.artifacts_dir / "result_manifest.json", [result.to_dict() for result in results])

    if bool(cfg.analysis.generate_figures):
        _generate_standard_figures(cfg, run_context, results, frame, logger)

    payload = {
        "results_frame": frame.to_dict(orient="records"),
        "summary_frame": summary.to_dict(orient="records"),
        "artifact_dir": str(run_context.artifacts_dir),
        "metrics_dir": str(run_context.metrics_dir),
    }
    write_json(run_context.artifacts_dir / "experiment_summary.json", payload)
    logger.info("Finished experiment '%s'; outputs written to %s", cfg.experiment.name, run_context.run_dir)
    return payload


def _generate_standard_figures(
    cfg: DictConfig,
    run_context: EmpiricalRunContext,
    results: Sequence[object],
    frame,
    logger: logging.Logger,
) -> None:
    metric = str(cfg.analysis.primary_metric)
    if not frame.empty:
        plot_metric_boxplot(frame, run_context.figures_dir / f"{metric}_boxplot.png", metric=metric)
        plot_metric_heatmap(frame, run_context.figures_dir / f"{metric}_heatmap.png", value_column=metric)
    for result in results:
        filename = f"{result.method}_seed_{result.seed}_trajectory.png"
        plot_loss_trajectory(result.history, run_context.figures_dir / filename, title=f"{cfg.experiment.name}: {result.method} seed {result.seed}")
    logger.info("Generated standard figures in %s", run_context.figures_dir)
