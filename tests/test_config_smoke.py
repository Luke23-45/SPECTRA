from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG_DIR = str((ROOT / "configs").resolve())
EMPIRICAL_CONFIG_DIR = str((ROOT / "configs" / "empirical").resolve())


def _compose(config_dir: str, config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides)


@pytest.mark.parametrize(
    ("overrides", "expected_dataset"),
    [
        ([], "synthetic"),
        (["method=bpgs"], "synthetic"),
        (["method=gradnorm_proxy"], "synthetic"),
        (["method=kendall"], "synthetic"),
        (["method=pcgrad"], "synthetic"),
        (["method=static"], "synthetic"),
        (["method=uwso"], "synthetic"),
        (["dataset=clinical", "method=bpgs"], "clinical"),
        (["dataset=yeast", "method=bpgs"], "yeast"),
        (["dataset=rf1", "method=kendall"], "rf1"),
        (["dataset=qm9", "method=static"], "qm9"),
        (["preset=nyuv2_bpgs"], "nyuv2"),
    ],
)
def test_training_configs_compose(overrides: list[str], expected_dataset: str) -> None:
    cfg = _compose(TRAINING_CONFIG_DIR, "config", overrides)
    assert cfg.dataset_name == expected_dataset
    assert cfg.method.name
    assert cfg.tasks
    assert cfg.train.batch_size > 0


@pytest.mark.parametrize(
    "experiment_name",
    [
        "exp_01_correlation_sweep",
        "exp_02_conflict_stability",
        "exp_03_imbalance_robustness",
        "exp_04_control_benign",
        "exp_05_multiclass_behavior",
        "exp_06_orchestrator",
        "exp_07_pure_loss_rescaling",
        "exp_08_heterogeneous_regime",
    ],
)
def test_empirical_configs_compose(experiment_name: str) -> None:
    cfg = _compose(EMPIRICAL_CONFIG_DIR, "config", [f"experiment={experiment_name}"])
    assert cfg.experiment.name == experiment_name
    assert cfg.family.name
    assert cfg.methods.names
    assert cfg.seeds.values
