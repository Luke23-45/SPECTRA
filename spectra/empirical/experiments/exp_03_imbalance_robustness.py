"""Imbalance robustness experiment."""

from __future__ import annotations

import logging
from typing import Dict

from omegaconf import DictConfig

from spectra.empirical.experiments.common import run_single_experiment
from spectra.empirical.utils import EmpiricalRunContext


def run(cfg: DictConfig, run_context: EmpiricalRunContext, logger: logging.Logger) -> Dict[str, object]:
    return run_single_experiment(cfg, run_context, logger)
