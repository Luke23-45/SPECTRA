"""Balanced control benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from spectra.empirical.generators.base import SyntheticGeneratorConfig


@dataclass(frozen=True)
class BalancedBenchmark:
    family: str = "correlation"
    generator: SyntheticGeneratorConfig = SyntheticGeneratorConfig(
        task_count=4,
        correlation=0.5,
        imbalance_ratio=1.0,
        label_noise=0.0,
        task_type_mix="mixed",
        n_samples=768,
    )
    methods: tuple[str, ...] = ("bpgs", "uwso", "static")
    epochs: int = 20
    hidden_dim: int = 48
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


def build_balanced_benchmark() -> BalancedBenchmark:
    return BalancedBenchmark()

