"""Multi-method comparison harness for synthetic benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from spectra.empirical.generators.base import SyntheticGeneratorConfig
from spectra.empirical.runners.base_runner import ExperimentResult, RunnerConfig
from spectra.empirical.runners.seed_manager import DEFAULT_PAIRED_SEEDS
from spectra.empirical.runners.synthetic_runner import SyntheticExperimentRunner, SyntheticRunSpec


@dataclass
class ComparisonConfig:
    family: str
    methods: Sequence[str]
    seeds: Sequence[int] = DEFAULT_PAIRED_SEEDS
    epochs: int = 20
    batch_size: int = 64
    hidden_dim: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    output_dir: str = "outputs/empirical"
    bpgs_architecture: Dict[str, object] | None = None
    loss_scale_task: str | None = None
    loss_scale_multiplier: float = 1.0
    logging: Dict[str, object] | None = None


class MethodComparisonHarness:
    """Run a family/configuration across methods and seeds."""

    def __init__(self, generator_config: SyntheticGeneratorConfig, comparison_config: ComparisonConfig) -> None:
        self.generator_config = generator_config
        self.comparison_config = comparison_config

    def run(self) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        for method in self.comparison_config.methods:
            for seed in self.comparison_config.seeds:
                run_spec = SyntheticRunSpec(
                    generator_config=SyntheticGeneratorConfig(**{**self.generator_config.to_dict(), "seed": int(seed)}),
                    runner_config=RunnerConfig(
                        family=self.comparison_config.family,
                        method=method,
                        seed=int(seed),
                        epochs=self.comparison_config.epochs,
                        batch_size=self.comparison_config.batch_size,
                        hidden_dim=self.comparison_config.hidden_dim,
                        learning_rate=self.comparison_config.learning_rate,
                        weight_decay=self.comparison_config.weight_decay,
                        device=self.comparison_config.device,
                        output_dir=self.comparison_config.output_dir,
                        bpgs_architecture=dict(self.comparison_config.bpgs_architecture or {}),
                        loss_scale_task=self.comparison_config.loss_scale_task,
                        loss_scale_multiplier=float(self.comparison_config.loss_scale_multiplier),
                        logging=dict(self.comparison_config.logging or {}),
                    ),
                )
                results.append(SyntheticExperimentRunner(run_spec).run())
        return results

    @staticmethod
    def to_frame(results: Sequence[ExperimentResult]) -> pd.DataFrame:
        rows = []
        for result in results:
            row = {"method": result.method, "seed": result.seed}
            row.update(result.final_metrics)
            rows.append(row)
        return pd.DataFrame(rows)
