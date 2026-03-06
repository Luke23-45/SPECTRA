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
    # Robust extraction supporting both old-style and new-style Hydra configs
    name = cfg.get("method_name") or cfg.get("method", {}).get("name")
    
    if name == "bpgs" or name == "bpgs_alb":
        from spectra.core.bpgs import BPGS
        m_cfg = cfg.get("method", {})
        
        # Tau fallback and ema_decay conversion
        tau = m_cfg.get("tau", 50.0)
        if "ema_decay" in m_cfg and "tau" not in m_cfg:
            import math
            decay = float(m_cfg.get("ema_decay"))
            decay_safe = max(1e-6, min(1.0 - 1e-6, decay))
            # SOTA TITANIUM FIX: Exact Discrete Integrator Formulation (Corrected)
            # Previously: -1.0 / math.log(1.0 - decay_safe) which maps 0.99 to 0.217 (instant forget)
            # Correctly derived constraint: beta = 1 - decay_safe
            # beta = 1 - exp(-1/tau) -> exp(-1/tau) = decay_safe -> -1/tau = ln(decay_safe) -> tau = -1/ln(decay_safe)
            tau = -1.0 / math.log(decay_safe)
            
        return BPGS(
            num_tasks=len(cfg.tasks),
            s_min=m_cfg.get("s_min", -10.0),
            s_max=m_cfg.get("s_max", 10.0),
            tau=tau,
            eps=m_cfg.get("eps", 1e-5),
            s_init=m_cfg.get("s_init", 0.0),
            prior_var=m_cfg.get("prior_var", None),
            use_autocal=m_cfg.get("use_autocal", False),
            autocal_margin=m_cfg.get("autocal_margin", 10.0)
        )
    
    cls = WEIGHTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown weighting method: {name}. Available: {list(WEIGHTER_REGISTRY.keys())}")
    
    params = cfg.get("params") or cfg.get("method", {}).get("params", {})
    return cls(num_tasks=len(cfg.tasks), **params)


__all__ = [
    "BaseWeighter", "StaticWeighter", "KendallWeighter",
    "UWSOWeighter", "PCGradWeighter", "NTKMTLWeighter",
    "WEIGHTER_REGISTRY", "build_weighter",
]
