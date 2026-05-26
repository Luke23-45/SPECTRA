"""Reliable Weights & Biases integration for SPECTRA training and studies."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


def _cfg_to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, DictConfig):
        resolved = OmegaConf.to_container(payload, resolve=True)
        return resolved if isinstance(resolved, dict) else {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _coerce_tags(value: Any) -> tuple[str, ...]:
    if value in (None, "", False):
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def flatten_metrics(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, prefix=full_key))
            continue
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[full_key] = float(value)
    return flat


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool = False
    project: str = "spectra-mtl"
    entity: Optional[str] = None
    mode: str = "offline"
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    notes: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    dir_name: str = "wandb"
    log_artifacts: bool = True
    log_config: bool = True
    log_model: bool = False

    @classmethod
    def from_logging_config(cls, logging_cfg: Any) -> "WandbSettings":
        logging_payload = _cfg_to_dict(logging_cfg)
        wandb_payload = _cfg_to_dict(logging_payload.get("wandb"))
        enabled = bool(
            wandb_payload.get(
                "enabled",
                logging_payload.get("use_wandb", False),
            )
        )
        mode = str(wandb_payload.get("mode", logging_payload.get("wandb_mode", "offline"))).lower()
        if mode in {"disabled", "off", "none"}:
            enabled = False
        return cls(
            enabled=enabled,
            project=str(wandb_payload.get("project", logging_payload.get("wandb_project", "spectra-mtl"))),
            entity=wandb_payload.get("entity"),
            mode=mode,
            name=wandb_payload.get("name"),
            group=wandb_payload.get("group"),
            job_type=wandb_payload.get("job_type"),
            notes=wandb_payload.get("notes"),
            tags=_coerce_tags(wandb_payload.get("tags")),
            dir_name=str(wandb_payload.get("dir_name", "wandb")),
            log_artifacts=bool(wandb_payload.get("log_artifacts", True)),
            log_config=bool(wandb_payload.get("log_config", True)),
            log_model=bool(wandb_payload.get("log_model", False)),
        )


class WandbSession:
    """Single-run W&B lifecycle manager bound to a SPECTRA-owned run directory."""

    def __init__(
        self,
        settings: WandbSettings,
        run_dir: Path,
        run_id: str,
        *,
        name: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Iterable[str] | None = None,
        config_payload: Mapping[str, Any] | None = None,
        summary_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.name = name or settings.name or self.run_id
        self.group = group or settings.group
        self.job_type = job_type or settings.job_type
        self.tags = tuple(settings.tags) + tuple(str(tag) for tag in (tags or ()))
        self.config_payload = dict(config_payload or {})
        self.summary_payload = dict(summary_payload or {})
        self.local_dir = self.run_dir / self.settings.dir_name
        self._run = None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def _ensure_dir(self) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        (self.local_dir / "cache").mkdir(parents=True, exist_ok=True)
        (self.local_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.local_dir / "config").mkdir(parents=True, exist_ok=True)
        return self.local_dir

    def _configure_environment(self) -> None:
        root = self._ensure_dir()
        os.environ["WANDB_DIR"] = str(root)
        os.environ["WANDB_CACHE_DIR"] = str(root / "cache")
        os.environ["WANDB_DATA_DIR"] = str(root / "data")
        os.environ["WANDB_CONFIG_DIR"] = str(root / "config")

    def _init_kwargs(self) -> Dict[str, Any]:
        self._configure_environment()
        return {
            "project": self.settings.project,
            "entity": self.settings.entity,
            "name": self.name,
            "id": self.run_id,
            "resume": "allow",
            "dir": str(self._ensure_dir()),
            "mode": self.settings.mode,
            "group": self.group,
            "job_type": self.job_type,
            "tags": list(dict.fromkeys(self.tags)) or None,
            "notes": self.settings.notes,
        }

    def create_lightning_logger(self) -> Optional[WandbLogger]:
        if not self.enabled:
            return None
        logger = WandbLogger(
            save_dir=str(self._ensure_dir()),
            dir=str(self._ensure_dir()),
            offline=(self.settings.mode == "offline"),
            log_model=self.settings.log_model,
            **{k: v for k, v in self._init_kwargs().items() if k != "dir"},
        )
        experiment = logger.experiment
        if experiment is not None:
            self._run = experiment
            if self.settings.log_config and self.config_payload:
                experiment.config.update(self.config_payload, allow_val_change=True)
            if self.summary_payload:
                for key, value in self.summary_payload.items():
                    experiment.summary[key] = value
        return logger

    def start(self) -> Optional[Any]:
        if not self.enabled:
            return None
        if self._run is not None:
            return self._run
        import wandb

        self._run = wandb.init(**self._init_kwargs())
        if self._run is not None:
            if self.settings.log_config and self.config_payload:
                self._run.config.update(self.config_payload, allow_val_change=True)
            if self.summary_payload:
                for key, value in self.summary_payload.items():
                    self._run.summary[key] = value
        return self._run

    def log_metrics(self, metrics: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        run = self.start()
        if run is None:
            return
        payload = flatten_metrics(metrics)
        if not payload:
            return
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=int(step))

    def update_summary(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        run = self.start()
        if run is None:
            return
        for key, value in payload.items():
            if isinstance(value, Mapping):
                for flat_key, flat_value in flatten_metrics(value, prefix=str(key)).items():
                    run.summary[flat_key] = flat_value
            else:
                run.summary[str(key)] = value

    def log_artifact(self, name: str, files: Mapping[str, Path], artifact_type: str = "run-artifacts") -> None:
        if not self.enabled or not self.settings.log_artifacts:
            return
        run = self.start()
        if run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        for alias, path in files.items():
            candidate = Path(path)
            if candidate.exists():
                artifact.add_file(str(candidate), name=alias)
        run.log_artifact(artifact)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        finally:
            self._run = None
