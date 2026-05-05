"""
Synthetic Data Generators

Three controlled families for isolated stress testing:
    - CorrelationFamily: Adjustable task correlation (-1 to 1)
    - ImbalanceFamily: Scale gaps and sample imbalance
    - ConflictFamily: Opposing gradients and anti-correlated targets
"""

__all__ = []
"""Synthetic data generators for empirical experiments."""

from spectra.empirical.generators.base import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
    SyntheticProblem,
    TaskSpec,
    TensorTaskDataset,
)
from spectra.empirical.generators.correlation_family import CorrelationControlledGenerator
from spectra.empirical.generators.imbalance_family import ImbalanceControlledGenerator
from spectra.empirical.generators.conflict_family import ConflictControlledGenerator
from spectra.empirical.generators.heterogeneous_family import HeterogeneousControlledGenerator
from spectra.empirical.generators.rescaling_family import RescalingControlledGenerator

__all__ = [
    "SyntheticGenerator",
    "SyntheticGeneratorConfig",
    "SyntheticProblem",
    "TaskSpec",
    "TensorTaskDataset",
    "CorrelationControlledGenerator",
    "ImbalanceControlledGenerator",
    "ConflictControlledGenerator",
    "RescalingControlledGenerator",
    "HeterogeneousControlledGenerator",
]
