"""
scripts/run_empirical_experiment.py
----------------------------------
Hydra entrypoint for empirical synthetic experiments.
"""

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import hydra
from omegaconf import DictConfig

from spectra.empirical.experiments import EXPERIMENT_REGISTRY
from spectra.empirical.utils import EmpiricalRunContext, validate_empirical_config


@hydra.main(config_path="../configs/empirical", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_empirical_config(cfg)
    run_context = EmpiricalRunContext.from_config(cfg).initialize()
    logger = run_context.setup_logging()
    run_context.save_latest_pointer()
    run_context.save_resolved_config(cfg)
    run_context.save_run_metadata(cfg)

    experiment_name = str(cfg.experiment.name)
    if experiment_name not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown empirical experiment: {experiment_name}")

    logger.info("Starting empirical experiment '%s'", experiment_name)
    EXPERIMENT_REGISTRY[experiment_name](cfg, run_context, logger)
    logger.info("Completed empirical experiment '%s'", experiment_name)


if __name__ == "__main__":
    main()
