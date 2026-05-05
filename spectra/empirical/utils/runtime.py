"""Runtime helpers for empirical experiments."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def _original_cwd() -> Path:
    try:
        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_original_cwd() / path).resolve()


def _git_info() -> Dict[str, Any]:
    info = {"commit_hash": "unknown", "branch": "unknown", "is_dirty": False}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_original_cwd(), capture_output=True, text=True)
        if result.returncode == 0:
            info["commit_hash"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_original_cwd(),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        result = subprocess.run(["git", "status", "--porcelain"], cwd=_original_cwd(), capture_output=True, text=True)
        if result.returncode == 0:
            info["is_dirty"] = bool(result.stdout.strip())
    except Exception:
        pass
    return info


def _system_info() -> Dict[str, Any]:
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system(),
        "platform_release": platform.release(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@dataclass(frozen=True)
class EmpiricalRunContext:
    """Resolved run-directory layout for one empirical experiment invocation."""

    experiment_name: str
    timestamp: str
    root_dir: Path
    run_dir: Path
    figures_dir: Path
    logs_dir: Path
    metrics_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "EmpiricalRunContext":
        experiment_name = str(cfg.experiment.name)
        timestamp = str(cfg.output.timestamp)
        root_dir = _resolve_path(str(cfg.output.root_dir))
        run_dir = root_dir / experiment_name / timestamp
        return cls(
            experiment_name=experiment_name,
            timestamp=timestamp,
            root_dir=root_dir,
            run_dir=run_dir,
            figures_dir=run_dir / "figures",
            logs_dir=run_dir / "logs",
            metrics_dir=run_dir / "metrics",
            artifacts_dir=run_dir / "artifacts",
        )

    def initialize(self) -> "EmpiricalRunContext":
        for directory in (self.run_dir, self.figures_dir, self.logs_dir, self.metrics_dir, self.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def latest_manifest_path(self) -> Path:
        return self.root_dir / self.experiment_name / "latest_run.json"

    def save_latest_pointer(self) -> Path:
        path = self.latest_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"experiment_name": self.experiment_name, "timestamp": self.timestamp, "run_dir": str(self.run_dir)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_resolved_config(self, cfg: DictConfig, filename: str = "config_resolved.yaml") -> Path:
        path = self.artifacts_dir / filename
        path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        return path

    def save_run_metadata(self, cfg: DictConfig, extra: Optional[Dict[str, Any]] = None) -> Path:
        path = self.artifacts_dir / "run_metadata.json"
        payload = {
            "experiment": {
                "name": self.experiment_name,
                "timestamp": self.timestamp,
                "run_dir": str(self.run_dir),
                "description": cfg.experiment.get("description"),
            },
            "system": _system_info(),
            "git": _git_info(),
            "config_summary": {
                "family": cfg.family.name,
                "methods": list(cfg.methods["names"]),
                "seed_values": list(cfg.seeds["values"]),
            },
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
        }
        if extra:
            payload["extra"] = extra
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def setup_logging(self, logger_name: str = "spectra.empirical") -> logging.Logger:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers = []
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        file_handler = logging.FileHandler(self.logs_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False
        return logger
