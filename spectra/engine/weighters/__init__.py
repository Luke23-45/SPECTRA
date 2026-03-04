"""
spectra/engine/weighters/__init__.py
------------------------------------
Centralized Multi-Task Learning Weighter Registry.
(Redirected to spectra.baselines to eliminate duplication).
"""

from spectra.baselines import (
    BaseWeighter,
    StaticWeighter,
    KendallWeighter,
    UWSOWeighter,
    PCGradWeighter,
    NTKMTLWeighter,
    build_weighter,
    WEIGHTER_REGISTRY
)

__all__ = [
    "BaseWeighter", "StaticWeighter", "KendallWeighter",
    "UWSOWeighter", "PCGradWeighter", "NTKMTLWeighter",
    "WEIGHTER_REGISTRY", "build_weighter",
]
