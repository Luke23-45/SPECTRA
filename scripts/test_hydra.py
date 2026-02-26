import os
import sys
from pathlib import Path

# --- NASA-Grade Path Resolution ---
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Hydra Success!")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
