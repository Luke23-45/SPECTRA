"""
spectra/data/nyuv2/__init__.py
-------------------------------
NYUv2 Multi-Task Dense Prediction Data Package.

Exports:
    - NYUv2Dataset: Core dataset (train/val loader)
    - NYUv2TrainTransform: Training augmentation pipeline
    - NYUv2TestTransform: Validation (no augmentation) pipeline
    - download_nyuv2: Automated data acquisition
    - NUM_CLASSES: 13
    - IGNORE_INDEX: 255
    - NYUv2_CLASS_NAMES: List of class names
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
from .download import download_nyuv2

__all__ = [
    "NYUv2Dataset",
    "NYUv2TrainTransform",
    "NYUv2TestTransform",
    "RandomScaleCrop",
    "RandomHorizontalFlip",
    "ImageNetNormalize",
    "download_nyuv2",
    "NUM_CLASSES",
    "IGNORE_INDEX",
    "NYUv2_CLASS_NAMES",
]
