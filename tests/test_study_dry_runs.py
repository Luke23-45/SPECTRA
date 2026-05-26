from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_study(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "experiments/bpgs_study/run.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ablation_dry_run_uses_current_experiments_subset_script() -> None:
    result = _run_study("--study", "01_nyuv2_ablation", "--dry-run")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "experiments\\bpgs_study\\nyuv2_subsets.py" in result.stdout
    assert "studies\\bpgs_study\\nyuv2_subsets.py" not in result.stdout


def test_real_data_dry_run_generates_training_commands() -> None:
    result = _run_study("--study", "03_yeast_regime_check", "--dry-run")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "dataset=yeast" in result.stdout
    assert "method=static" in result.stdout
    assert "method=pcgrad" in result.stdout


def test_stress_dry_run_generates_empirical_commands() -> None:
    result = _run_study("--study", "02_synthetic_scale_stress", "--dry-run")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "experiment=exp_03_imbalance_robustness" in result.stdout
    assert "family=imbalance" in result.stdout
    assert "methods.names=[bpgs]" in result.stdout
