"""
Experiment Runners

Execution and orchestration:
    - base_runner: Abstract experiment runner
    - synthetic_runner: Synthetic suite execution
    - method_comparison: Multi-method comparison harness
    - seed_manager: Paired seed reproducibility (3 seeds)
"""

__all__ = []
"""Experiment runner interfaces."""

from spectra.empirical.runners.base_runner import BaseRunner, ExperimentResult, RunnerConfig
from spectra.empirical.runners.method_comparison import ComparisonConfig, MethodComparisonHarness
from spectra.empirical.runners.seed_manager import DEFAULT_PAIRED_SEEDS, SeedBundle, apply_seed_bundle, paired_seeds
from spectra.empirical.runners.synthetic_runner import SyntheticExperimentRunner, SyntheticRunSpec

__all__ = [
    "BaseRunner",
    "ExperimentResult",
    "RunnerConfig",
    "ComparisonConfig",
    "MethodComparisonHarness",
    "DEFAULT_PAIRED_SEEDS",
    "SeedBundle",
    "apply_seed_bundle",
    "paired_seeds",
    "SyntheticExperimentRunner",
    "SyntheticRunSpec",
]
