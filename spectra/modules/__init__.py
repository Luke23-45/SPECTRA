"""
spectra/modules/__init__.py
---------------------------
Exposes the fully decoupled Vertical Silos for Hydra deployment.
"""

from spectra.modules.base import OrthogonalSPECTRAModule
from spectra.modules.clinical import ClinicalSPECTRAModule
from spectra.modules.vision import VisionSPECTRAModule
from spectra.modules.synthetic import SyntheticSPECTRAModule

__all__ = [
    "OrthogonalSPECTRAModule",
    "ClinicalSPECTRAModule", 
    "VisionSPECTRAModule", 
    "SyntheticSPECTRAModule"
]
