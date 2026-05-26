"""
Configuration loader for the BPGS analysis framework.

Loads ``configs/base.yaml`` via OmegaConf and auto-resolves the
``paths.experiment_root`` and ``paths.data_root`` values based on
the location of this file on disk.

Usage::

    from analysis.common.config import get_config

    cfg = get_config()
    data_root = cfg.paths.data_root
    ablation_dir = Path(cfg.paths.data_root) / cfg.paths.ablation

Override at runtime via OmegaConf merge::

    from analysis.common.config import get_config
    cfg = get_config(overrides={"paths.data_root": "/custom/path"})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[DictConfig] = None
_CONFIGS_DIR: Path = Path(__file__).resolve().parent.parent / "configs"
_DEFAULT_CONFIG: Path = _CONFIGS_DIR / "base.yaml"


def _detect_experiment_root() -> Path:
    """Walk up from this file to find the bpgs_experiment/ root.

    Layout: analysis/common/config.py → common/ → analysis/ → bpgs_experiment/
    """
    return Path(__file__).resolve().parent.parent.parent


def _detect_data_root() -> Path:
    """Resolve the bpgs_experiment/data/ directory."""
    return _detect_experiment_root() / "data"


def get_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    use_cache: bool = True,
) -> DictConfig:
    """Load and return the analysis configuration.

    On first call, loads ``configs/base.yaml``, sets auto-detected
    ``paths.experiment_root`` and ``paths.data_root``, resolves all
    OmegaConf interpolations, and caches the result.

    Parameters
    ----------
    config_path : Path, optional
        Path to a YAML config file.  Defaults to ``configs/base.yaml``.
    overrides : dict, optional
        Key-value pairs to merge on top of the loaded config.
        Dot-notation keys are supported (e.g. ``{"paths.data_root": "..."}``).
    use_cache : bool
        If True (default), return the cached config on subsequent calls.
        Set to False to force a fresh load (useful in tests).

    Returns
    -------
    DictConfig
        Fully resolved OmegaConf configuration.
    """
    global _CONFIG_CACHE

    if use_cache and _CONFIG_CACHE is not None and overrides is None:
        return _CONFIG_CACHE

    cfg_path = config_path or _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Analysis config not found: {cfg_path}. "
            f"Expected at: {_DEFAULT_CONFIG}"
        )

    cfg = OmegaConf.load(cfg_path)
    assert isinstance(cfg, DictConfig)

    if cfg.paths.experiment_root is None:
        experiment_root = _detect_experiment_root()
        OmegaConf.update(cfg, "paths.experiment_root", str(experiment_root))
        logger.debug("Auto-detected experiment_root: %s", experiment_root)

    if cfg.paths.data_root is None:
        data_root = _detect_data_root()
        OmegaConf.update(cfg, "paths.data_root", str(data_root))
        logger.debug("Auto-detected data_root: %s", data_root)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in overrides.items()]
        )
        cfg = OmegaConf.merge(cfg, override_cfg)

    if overrides is None:
        _CONFIG_CACHE = cfg

    logger.info(
        "Config loaded: data_root=%s",
        OmegaConf.select(cfg, "paths.data_root"),
    )
    return cfg


def clear_config_cache() -> None:
    """Clear the cached configuration (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def get_data_root(cfg: Optional[DictConfig] = None) -> Path:
    """Return the resolved data root as a Path object.

    Parameters
    ----------
    cfg : DictConfig, optional
        Pre-loaded config.  Loads default if None.

    Returns
    -------
    Path
        Absolute path to ``bpgs_experiment/data/``.
    """
    if cfg is None:
        cfg = get_config()
    return Path(cfg.paths.data_root)


def get_study_data_dir(
    study: str,
    cfg: Optional[DictConfig] = None,
) -> Path:
    """Return the absolute data directory for a specific study.

    Parameters
    ----------
    study : str
        One of: ``"ablation"``, ``"nyuv2"``, ``"yeast"``, ``"rf1"``,
        ``"stress_scale"``, ``"stress_rescale"``, ``"stress_hetero"``.
    cfg : DictConfig, optional
        Pre-loaded config.

    Returns
    -------
    Path
        Absolute path to the study's data directory.

    Raises
    ------
    KeyError
        If the study name is not found in ``paths``.
    """
    if cfg is None:
        cfg = get_config()

    data_root = Path(cfg.paths.data_root)
    relative_dir = OmegaConf.select(cfg, f"paths.{study}")

    if relative_dir is None:
        available = [
            k for k in cfg.paths
            if k not in ("experiment_root", "data_root", "output_root")
        ]
        raise KeyError(
            f"Unknown study '{study}'. "
            f"Available studies in config: {available}"
        )

    return data_root / relative_dir


def get_output_root(cfg: Optional[DictConfig] = None) -> Path:
    """Return the resolved output root as a Path object.

    Parameters
    ----------
    cfg : DictConfig, optional
        Pre-loaded config.

    Returns
    -------
    Path
        Absolute path to the analysis output directory.
    """
    if cfg is None:
        cfg = get_config()
    experiment_root = Path(cfg.paths.experiment_root)
    return experiment_root / cfg.paths.output_root


def get_seeds(cfg: Optional[DictConfig] = None) -> List[int]:
    """Return the canonical seed values from config.

    Parameters
    ----------
    cfg : DictConfig, optional
        Pre-loaded config.

    Returns
    -------
    list of int
    """
    if cfg is None:
        cfg = get_config()
    return list(cfg.seeds)


def get_methods(
    study: str,
    cfg: Optional[DictConfig] = None,
) -> List[str]:
    """Return the method list for a study from config.

    Parameters
    ----------
    study : str
        Study identifier.  Supports dot-notation for nested studies
        like ``"full_data.yeast"`` or ``"stress.heterogeneous"``.
    cfg : DictConfig, optional
        Pre-loaded config.

    Returns
    -------
    list of str
    """
    if cfg is None:
        cfg = get_config()
    methods = OmegaConf.select(cfg, f"studies.{study}.methods")
    if methods is None:
        raise KeyError(
            f"No methods defined for study '{study}' in config."
        )
    return list(methods)
