# /tmp/debug_config.py
import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def debug(cfg: DictConfig):
    print("--- Resolved Config ---")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    debug()
