from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_FILES = (
    "02_synthetic_scale_stress.yaml",
    "06_pure_loss_rescaling.yaml",
    "07_heterogeneous_mixed_stress.yaml",
)
