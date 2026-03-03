"""
spectra/utils/config.py
-----------------------
Universal configuration merging and routing utilities.
"""
from omegaconf import DictConfig, OmegaConf

def _merge_dataset_defaults(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """
    Merge dataset defaults with top-level overrides without struct-key crashes.

    OmegaConf structured nodes can reject unknown keys during direct merge
    (`ConfigKeyError`). We convert both to plain dict first, then recreate a
    DictConfig so CLI/top-level keys (e.g. `model.n_heads`) are preserved.
    """
    cli_overrides = OmegaConf.to_container(base_cfg, resolve=False) if base_cfg is not None else {}
    defaults = OmegaConf.to_container(override_cfg, resolve=False) if override_cfg is not None else {}
    if not isinstance(cli_overrides, dict):
        cli_overrides = {}
    if not isinstance(defaults, dict):
        defaults = {}
    
    # [SOTA FIX]: Right overwrites Left! So defaults MUST be on the left, 
    # and cli_overrides on the right to preserve CLI intent!
    merged = OmegaConf.merge(defaults, cli_overrides)
    return OmegaConf.create(merged)
