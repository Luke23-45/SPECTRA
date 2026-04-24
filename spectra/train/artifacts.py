"""
spectra/train/artifacts.py
--------------------------
Stable artifact and resume-path utilities for long-running training jobs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def _original_cwd() -> Path:
    try:
        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def resolve_artifact_dir(cfg: DictConfig) -> Path:
    """Resolve the stable run artifact directory outside Hydra timestamp shells."""
    run_name = cfg.get("run_name", "unnamed_run")
    rel_path = cfg.get("output_dir", f"./outputs/{run_name}")
    return (_original_cwd() / Path(rel_path)).resolve()


def resolve_resume_checkpoint(cfg: DictConfig, artifact_dir: Path) -> Optional[Path]:
    """
    Resolve the checkpoint path for resume.

    Supported values:
      - null / empty: no resume
      - "auto": resume from `<artifact_dir>/checkpoints/last.ckpt` if present
      - explicit path: absolute or relative to the original working directory
    """
    raw_resume = cfg.get("resume_from", None)
    if raw_resume in (None, "", False):
        return None

    resume_str = str(raw_resume).strip()
    if not resume_str:
        return None

    if resume_str.lower() == "auto":
        candidate = artifact_dir / "checkpoints" / "last.ckpt"
        return candidate if candidate.exists() else None

    candidate = Path(resume_str)
    if not candidate.is_absolute():
        candidate = (_original_cwd() / candidate).resolve()
    return candidate


def stable_run_id(cfg: DictConfig) -> str:
    """Stable logger/run identifier safe for WandB resume."""
    run_name = str(cfg.get("run_name", "unnamed_run"))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")
    return safe or "unnamed-run"
