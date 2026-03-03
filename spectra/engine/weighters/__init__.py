"""
spectra/engine/weighters/__init__.py
------------------------------------
Centralized Multi-Task Learning Weighter Registry.
"""

from spectra.engine.weighters.base import BaseWeighter
from spectra.engine.weighters.static import StaticWeighter
from spectra.engine.weighters.kendall import KendallWeighter
from spectra.engine.weighters.uwso import UWSOWeighter
from spectra.engine.weighters.pcgrad import PCGradWeighter
from spectra.engine.weighters.ntkmtl import NTKMTLWeighter

WEIGHTER_REGISTRY = {
    "static": StaticWeighter,
    "kendall": KendallWeighter,
    "uwso": UWSOWeighter,
    "pcgrad": PCGradWeighter,
    "ntkmtl": NTKMTLWeighter,
}

def build_weighter(cfg):
    """Factory: builds a weighter from Hydra config."""
    # Robust extraction of weighting method name
    name = cfg.get("method_name") or cfg.get("method", {}).get("name")
    
    if name == "bpgs" or name == "bpgs_alb":
        from spectra.core.bpgs import BPGSScaler
        return BPGSScaler(
            num_tasks=len(cfg.tasks),
            s_min=cfg.get("s_min") or cfg.get("method", {}).get("s_min", -2.0),
            s_max=cfg.get("s_max") or cfg.get("method", {}).get("s_max", 10.0),
            ema_decay=cfg.get("ema_decay") or cfg.get("method", {}).get("ema_decay", 0.99),
            use_sigmoid=cfg.get("use_sigmoid") if cfg.get("use_sigmoid") is not None else cfg.get("method", {}).get("use_sigmoid", True),
            use_autocal=cfg.get("use_autocal") if cfg.get("use_autocal") is not None else cfg.get("method", {}).get("use_autocal", True),
        )
    
    cls = WEIGHTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown weighting method: {name}. Available: {list(WEIGHTER_REGISTRY.keys())}")
    
    params = cfg.get("params") or cfg.get("method", {}).get("params", {})
    return cls(num_tasks=len(cfg.tasks), **params)
