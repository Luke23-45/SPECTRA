"""
scripts/run_training.py
-----------------------
Mission Control for SPECTRA Experiments.
Sterile Entry Point — all orchestration is delegated to `spectra.train.runner`.
"""

import sys
from pathlib import Path

# --- Path Resolution ---
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from spectra.train.runner import execute_training_mission

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    execute_training_mission(cfg, output_dir)

if __name__ == "__main__":
    main()
