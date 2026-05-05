"""Imbalanced stress benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from spectra.empirical.generators.base import SyntheticGeneratorConfig


@dataclass(frozen=True)
class ImbalancedBenchmark:
    family: str = "imbalance"
    generator: SyntheticGeneratorConfig = SyntheticGeneratorConfig(
        task_count=4,
        correlation=0.15,
        imbalance_ratio=10.0,
        label_noise=0.1,
        task_type_mix="mixed",
        n_samples=960,
    )
    methods: tuple[str, ...] = ("bpgs", "uwso", "static")
    epochs: int = 24
    hidden_dim: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "generator": self.generator.to_dict(),
            "methods": list(self.methods),
            "epochs": self.epochs,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


def build_imbalanced_benchmark() -> ImbalancedBenchmark:
    return ImbalancedBenchmark()

