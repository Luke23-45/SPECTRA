"""
Executable Experiments

Six focused experiments + orchestrator:
    - exp_01_correlation_sweep: Boundedness validation
    - exp_02_conflict_stability: Stability under conflict
    - exp_03_imbalance_robustness: Imbalance stress tests
    - exp_04_control_benign: Control case (aligned tasks)
    - exp_05_multiclass_behavior: Mixed task types
    - exp_06_orchestrator: Parameterized orchestration
"""

__all__ = []
"""Executable empirical experiment entrypoints."""

from spectra.empirical.experiments.exp_01_correlation_sweep import run as run_exp_01
from spectra.empirical.experiments.exp_02_conflict_stability import run as run_exp_02
from spectra.empirical.experiments.exp_03_imbalance_robustness import run as run_exp_03
from spectra.empirical.experiments.exp_04_control_benign import run as run_exp_04
from spectra.empirical.experiments.exp_05_multiclass_behavior import run as run_exp_05
from spectra.empirical.experiments.exp_06_orchestrator import run as run_exp_06
from spectra.empirical.experiments.exp_07_pure_loss_rescaling import run as run_exp_07
from spectra.empirical.experiments.exp_08_heterogeneous_regime import run as run_exp_08

EXPERIMENT_REGISTRY = {
    "exp_01_correlation_sweep": run_exp_01,
    "exp_02_conflict_stability": run_exp_02,
    "exp_03_imbalance_robustness": run_exp_03,
    "exp_04_control_benign": run_exp_04,
    "exp_05_multiclass_behavior": run_exp_05,
    "exp_06_orchestrator": run_exp_06,
    "exp_07_pure_loss_rescaling": run_exp_07,
    "exp_08_heterogeneous_regime": run_exp_08,
}

__all__ = ["EXPERIMENT_REGISTRY"]
