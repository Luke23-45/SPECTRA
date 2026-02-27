"""
spectra/data/nyuv2/__init__.py
-------------------------------
NYUv2 Multi-Task Dense Prediction Data Package.

Exports:
    - NYUv2Dataset: Core dataset (train/val loader)
    - NYUv2TrainTransform: Training augmentation pipeline
    - NYUv2TestTransform: Validation (no augmentation) pipeline
"""

from .dataset import (
    NYUv2Dataset,
    NUM_CLASSES,
    IGNORE_INDEX,
    NYUv2_CLASS_NAMES,
)
from .transforms import (
    NYUv2TrainTransform,
    NYUv2TestTransform,
    RandomScaleCrop,
    RandomHorizontalFlip,
    ImageNetNormalize,
)

__all__ = [
    "NYUv2Dataset",
    "NYUv2TrainTransform",
    "NYUv2TestTransform",
    "RandomScaleCrop",
    "RandomHorizontalFlip",
    "ImageNetNormalize",
    "NUM_CLASSES",
    "IGNORE_INDEX",
    "NYUv2_CLASS_NAMES",
]
