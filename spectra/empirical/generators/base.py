"""Core abstractions for empirical synthetic benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


TaskType = Literal["regression", "binary", "multiclass", "angular"]
TaskMix = Literal["regression", "classification", "mixed"]


@dataclass(frozen=True)
class TaskSpec:
    """Structured task definition used across generation, training, and evaluation."""

    name: str
    task_type: TaskType
    loss: str
    output_dim: int = 1
    scale: float = 1.0
    offset: float = 0.0
    num_classes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TaskSpec.name must be non-empty.")
        if self.task_type not in {"regression", "binary", "multiclass", "angular"}:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if self.scale <= 0:
            raise ValueError(f"TaskSpec.scale must be positive; got {self.scale}.")
        if self.task_type == "multiclass":
            if self.num_classes is None or self.num_classes < 2:
                raise ValueError("Multiclass tasks require num_classes >= 2.")
            if self.output_dim != self.num_classes:
                raise ValueError("Multiclass task output_dim must equal num_classes.")
        elif self.task_type == "binary":
            if self.output_dim != 1:
                raise ValueError("binary tasks must use output_dim=1.")
        elif self.task_type == "regression":
            if self.output_dim < 1:
                raise ValueError("regression tasks must use output_dim >= 1.")
        elif self.task_type == "angular":
            if self.output_dim < 2:
                raise ValueError("angular tasks must use output_dim >= 2.")

    @property
    def is_classification(self) -> bool:
        return self.task_type in {"binary", "multiclass"}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticGeneratorConfig:
    """Validated configuration for controlled synthetic problem generation."""

    n_samples: int = 1024
    input_dim: int = 20
    shared_dim: int = 16
    task_count: int = 4
    correlation: float = 0.0
    imbalance_ratio: float = 1.0
    label_noise: float = 0.0
    task_type_mix: TaskMix = "mixed"
    train_fraction: float = 0.8
    seed: int = 42
    batch_size: int = 64
    regression_bias_scale: float = 0.25
    classification_temperature: float = 1.0
    multiclass_num_classes: int = 3

    def __post_init__(self) -> None:
        if self.n_samples < 32:
            raise ValueError("n_samples must be at least 32.")
        if self.input_dim < 2 or self.shared_dim < 2:
            raise ValueError("input_dim and shared_dim must both be >= 2.")
        if self.task_count < 2:
            raise ValueError("task_count must be at least 2.")
        if not -0.99 <= self.correlation <= 0.99:
            raise ValueError("correlation must lie in [-0.99, 0.99].")
        if self.imbalance_ratio < 1.0:
            raise ValueError("imbalance_ratio must be >= 1.")
        if not 0.0 <= self.label_noise <= 0.49:
            raise ValueError("label_noise must lie in [0.0, 0.49].")
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in [0.5, 1.0).")
        if self.batch_size < 4:
            raise ValueError("batch_size must be at least 4.")
        if self.classification_temperature <= 0:
            raise ValueError("classification_temperature must be positive.")
        if self.multiclass_num_classes < 3:
            raise ValueError("multiclass_num_classes must be at least 3.")

    @property
    def val_size(self) -> int:
        return self.n_samples - self.train_size

    @property
    def train_size(self) -> int:
        return int(round(self.n_samples * self.train_fraction))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TensorTaskDataset(Dataset):
    """Dataset wrapper exposing features and nested target dictionaries."""

    def __init__(self, inputs: torch.Tensor, targets: Mapping[str, torch.Tensor]) -> None:
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [N, D].")
        if not targets:
            raise ValueError("targets must be non-empty.")
        target_lengths = {name: tensor.shape[0] for name, tensor in targets.items()}
        if len(set(target_lengths.values())) != 1 or next(iter(target_lengths.values())) != inputs.shape[0]:
            raise ValueError("All targets must share the same leading dimension as inputs.")
        self.inputs = inputs
        self.targets = dict(targets)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "input": self.inputs[index],
            "targets": {name: tensor[index] for name, tensor in self.targets.items()},
        }

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        inputs = torch.stack([item["input"] for item in batch], dim=0)
        target_names = list(batch[0]["targets"].keys())
        targets = {name: torch.stack([item["targets"][name] for item in batch], dim=0) for name in target_names}
        return {"input": inputs, "targets": targets}


@dataclass
class SyntheticProblem:
    """Full generated problem instance, including splits and metadata."""

    train_dataset: TensorTaskDataset
    val_dataset: TensorTaskDataset
    task_specs: List[TaskSpec]
    config: SyntheticGeneratorConfig
    family: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_names(self) -> List[str]:
        return [task.name for task in self.task_specs]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "config": self.config.to_dict(),
            "task_specs": [task.to_dict() for task in self.task_specs],
            "metadata": self.metadata,
        }


class SyntheticGenerator(ABC):
    """Base class for deterministic synthetic benchmark generators."""

    family_name = "base"

    def __init__(self, config: SyntheticGeneratorConfig) -> None:
        self.config = config
        self._generator = torch.Generator().manual_seed(config.seed)

    def generate(self) -> SyntheticProblem:
        features, targets, task_specs, metadata = self._generate_core()
        if features.shape != (self.config.n_samples, self.config.input_dim):
            raise ValueError("Generator returned invalid feature shape.")
        indices = torch.randperm(self.config.n_samples, generator=self._generator)
        train_idx = indices[: self.config.train_size]
        val_idx = indices[self.config.train_size :]
        train_dataset = TensorTaskDataset(features[train_idx], {k: v[train_idx] for k, v in targets.items()})
        val_dataset = TensorTaskDataset(features[val_idx], {k: v[val_idx] for k, v in targets.items()})
        return SyntheticProblem(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            task_specs=task_specs,
            config=self.config,
            family=self.family_name,
            metadata=metadata,
        )

    @abstractmethod
    def _generate_core(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[TaskSpec], Dict[str, Any]]:
        """Return features, target tensors, task specifications, and metadata."""

    def _make_task_specs(self) -> List[TaskSpec]:
        specs: List[TaskSpec] = []
        for index in range(self.config.task_count):
            if self.config.task_type_mix == "regression":
                task_type = "regression"
            elif self.config.task_type_mix == "classification":
                task_type = "binary"
            else:
                if index % 3 == 0:
                    task_type = "multiclass"
                elif index % 2 == 0:
                    task_type = "binary"
                else:
                    task_type = "regression"

            if task_type == "regression":
                specs.append(TaskSpec(name=f"task_{index}", task_type="regression", loss="mse"))
            elif task_type == "binary":
                specs.append(TaskSpec(name=f"task_{index}", task_type="binary", loss="bce"))
            else:
                num_classes = self.config.multiclass_num_classes
                specs.append(
                    TaskSpec(
                        name=f"task_{index}",
                        task_type="multiclass",
                        loss="cross_entropy",
                        output_dim=num_classes,
                        num_classes=num_classes,
                    )
                )
        return specs

    def _sample_representation(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.config.n_samples, self.config.input_dim, generator=self._generator)
        w1 = torch.randn(self.config.input_dim, self.config.shared_dim, generator=self._generator) / self.config.input_dim**0.5
        w2 = torch.randn(self.config.input_dim, self.config.shared_dim, generator=self._generator) / self.config.input_dim**0.5
        z = torch.tanh(x @ w1) + 0.5 * torch.sin(x @ w2)
        z = z / z.std(dim=0, keepdim=True).clamp(min=1e-6)
        return x, z, w1

    def _correlated_vectors(self, reference: torch.Tensor, count: int, correlation: float) -> List[torch.Tensor]:
        reference = reference / reference.norm().clamp(min=1e-6)
        vectors: List[torch.Tensor] = []
        for _ in range(count):
            noise = torch.randn(reference.shape, generator=self._generator, dtype=reference.dtype, device=reference.device)
            noise = noise - torch.dot(noise, reference) * reference
            noise = noise / noise.norm().clamp(min=1e-6)
            vectors.append(correlation * reference + (1.0 - correlation**2) ** 0.5 * noise)
        return vectors

    def _noise(self, shape: Iterable[int], scale: float = 1.0) -> torch.Tensor:
        return torch.randn(*tuple(shape), generator=self._generator) * scale

    def _flip_binary_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.config.label_noise <= 0:
            return labels
        mask = torch.rand(labels.shape, generator=self._generator) < self.config.label_noise
        flipped = labels.clone()
        flipped[mask] = 1.0 - flipped[mask]
        return flipped

    def _flip_multiclass_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        if self.config.label_noise <= 0:
            return labels
        mask = torch.rand(labels.shape, generator=self._generator) < self.config.label_noise
        replacement = torch.randint(0, num_classes, labels.shape, generator=self._generator)
        return torch.where(mask, replacement, labels)
