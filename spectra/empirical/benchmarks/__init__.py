"""
Benchmark Configurations

Predefined stress settings for reproducible experiments:
    - config_balanced: Baseline, no stress
    - config_imbalanced: Extreme scale gaps
    - config_conflicting: Gradient conflict
    - sweeps: Grid search configurations (paper + full)
"""

__all__ = []
"""Benchmark presets and sweep utilities."""

from spectra.empirical.benchmarks.config_balanced import BalancedBenchmark, build_balanced_benchmark
from spectra.empirical.benchmarks.config_conflicting import ConflictingBenchmark, build_conflicting_benchmark
from spectra.empirical.benchmarks.config_imbalanced import ImbalancedBenchmark, build_imbalanced_benchmark
from spectra.empirical.benchmarks.sweeps import FULL_SWEEP_CONFIG, PAPER_CONFIG, SweepPoint, iter_sweep, materialize_sweep

__all__ = [
    "BalancedBenchmark",
    "ConflictingBenchmark",
    "ImbalancedBenchmark",
    "build_balanced_benchmark",
    "build_conflicting_benchmark",
    "build_imbalanced_benchmark",
    "PAPER_CONFIG",
    "FULL_SWEEP_CONFIG",
    "SweepPoint",
    "iter_sweep",
    "materialize_sweep",
]
