"""Base runner infrastructure for empirical experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import json
import pandas as pd


@dataclass
class RunnerConfig:
    family: str
    method: str
    seed: int
    epochs: int = 20
    batch_size: int = 64
    hidden_dim: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    device: str = "cpu"
    output_dir: str = "outputs/empirical"
    bpgs_architecture: Dict[str, Any] = field(default_factory=dict)
    loss_scale_task: str | None = None
    loss_scale_multiplier: float = 1.0
    logging: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    problem_metadata: Dict[str, Any]
    final_metrics: Dict[str, float]
    task_metrics: Dict[str, Dict[str, float]]
    history: List[Dict[str, Any]]
    method: str
    seed: int
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseRunner(ABC):
    """Shared serialization and lifecycle support for empirical runners."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir = self._result_dir()

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Execute the experiment and return the full result object."""

    def _result_dir(self) -> Path:
        path = self.output_dir / self.config.family / self.config.method / f"seed_{self.config.seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_result(self, result: ExperimentResult) -> ExperimentResult:
        json_path = self.result_dir / "result.json"
        csv_path = self.result_dir / "history.csv"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
        pd.DataFrame(result.history).to_csv(csv_path, index=False)
        result.artifacts["result_json"] = str(json_path)
        result.artifacts["history_csv"] = str(csv_path)
        return result
