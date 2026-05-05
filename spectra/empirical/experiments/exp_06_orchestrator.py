"""Parameterized orchestration for the empirical benchmark suite."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from spectra.empirical.analysis import summarize_comparison_frame
from spectra.empirical.experiments.common import run_single_experiment
from spectra.empirical.utils import EmpiricalRunContext, write_frame_csv, write_frame_json, write_jsonl


def run(cfg: DictConfig, run_context: EmpiricalRunContext, logger: logging.Logger) -> Dict[str, object]:
    if not cfg.experiment.get("sweep", {}).get("enabled", False):
        return run_single_experiment(cfg, run_context, logger)

    sweep_rows: List[Dict[str, object]] = []
    sweep_cfgs = cfg.experiment.sweep.parameters
    for task_count in sweep_cfgs.task_count:
        for correlation in sweep_cfgs.correlation:
            for imbalance_ratio in sweep_cfgs.imbalance_ratio:
                for label_noise in sweep_cfgs.label_noise:
                    variant = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                    variant.family.name = (
                        "conflict"
                        if correlation < 0
                        else ("imbalance" if imbalance_ratio > 1 else "correlation")
                    )
                    variant.experiment.generator.task_count = task_count
                    variant.experiment.generator.correlation = correlation
                    variant.experiment.generator.imbalance_ratio = imbalance_ratio
                    variant.experiment.generator.label_noise = label_noise
                    variant.output.timestamp = f"{cfg.output.timestamp}_tc{task_count}_corr{correlation}_imb{imbalance_ratio}_noise{label_noise}"
                    variant_context = EmpiricalRunContext.from_config(variant).initialize()
                    variant_context.save_latest_pointer()
                    variant_context.save_resolved_config(variant)
                    variant_context.save_run_metadata(variant, extra={"parent_run_dir": str(run_context.run_dir)})
                    payload = run_single_experiment(variant, variant_context, logger)
                    for row in payload["results_frame"]:
                        row["sweep_task_count"] = task_count
                        row["sweep_correlation"] = correlation
                        row["sweep_imbalance_ratio"] = imbalance_ratio
                        row["sweep_label_noise"] = label_noise
                        sweep_rows.append(row)

    frame = pd.DataFrame(sweep_rows)
    summary = summarize_comparison_frame(frame, metric=str(cfg.analysis.primary_metric))
    write_frame_csv(run_context.metrics_dir / "sweep_results.csv", frame)
    write_frame_json(run_context.metrics_dir / "sweep_results.json", frame)
    write_jsonl(run_context.metrics_dir / "sweep_results.jsonl", frame.to_dict(orient="records"))
    write_frame_csv(run_context.metrics_dir / "sweep_summary.csv", summary)
    write_frame_json(run_context.metrics_dir / "sweep_summary.json", summary)
    return {
        "results_frame": frame.to_dict(orient="records"),
        "summary_frame": summary.to_dict(orient="records"),
    }
