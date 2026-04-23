"""
spectra/baselines/ntkmtl.py
---------------------------
Legacy compatibility shim.

The active honest proxy baseline is `gradnorm_proxy`, not `ntkmtl`.
"""

from spectra.legacy.baselines.ntkmtl_engineered import NTKMTLWeighter

__all__ = ["NTKMTLWeighter"]
