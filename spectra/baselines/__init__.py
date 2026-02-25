"""SPECTRA Baselines: All MTL weighting methods for comparison."""

from spectra.baselines.base import BaseWeighter
from spectra.baselines.static import StaticWeighter
from spectra.baselines.kendall import KendallWeighter
from spectra.baselines.uwso import UWSOWeighter
from spectra.baselines.pcgrad import PCGradWeighter
from spectra.baselines.ntkmtl import NTKMTLWeighter

WEIGHTER_REGISTRY = {
    "static": StaticWeighter,
    "kendall": KendallWeighter,
    "uwso": UWSOWeighter,
    "pcgrad": PCGradWeighter,
    "ntkmtl": NTKMTLWeighter,
    "bpgs": None,  # Registered separately from core
}


def build_weighter(cfg):
    """Factory: builds a weighter from Hydra config."""
    name = cfg.method.name
    if name == "bpgs" or name == "bpgs_alb":
        from spectra.core.bpgs import BPGSScaler
        return BPGSScaler(
            num_tasks=len(cfg.tasks),
            s_min=cfg.method.get("s_min", -2.0),
            s_max=cfg.method.get("s_max", 10.0),
            ema_decay=cfg.method.get("ema_decay", 0.99),
            use_sigmoid=cfg.method.get("use_sigmoid", True),
            use_autocal=cfg.method.get("use_autocal", True),
        )
    
    cls = WEIGHTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown weighting method: {name}. Available: {list(WEIGHTER_REGISTRY.keys())}")
    
    params = cfg.method.get("params", {})
    return cls(num_tasks=len(cfg.tasks), **params)


__all__ = [
    "BaseWeighter", "StaticWeighter", "KendallWeighter",
    "UWSOWeighter", "PCGradWeighter", "NTKMTLWeighter",
    "WEIGHTER_REGISTRY", "build_weighter",
]
