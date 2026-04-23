"""SPECTRA baselines: single active canonical method path."""

from spectra.baselines.base import BaseWeighter
from spectra.baselines.static import StaticWeighter
from spectra.baselines.kendall import KendallWeighter
from spectra.baselines.uwso import UWSOWeighter
from spectra.baselines.pcgrad import PCGradWeighter
from spectra.baselines.gradnorm_proxy import GradNormProxyWeighter


WEIGHTER_REGISTRY = {
    "static": StaticWeighter,
    "kendall": KendallWeighter,
    "uwso": UWSOWeighter,
    "pcgrad": PCGradWeighter,
    "gradnorm_proxy": GradNormProxyWeighter,
    "bpgs": None,
    "bpgs_alb": None,
}


def build_weighter(cfg):
    """Factory: build a weighter from config."""
    name = cfg.get("method_name") or cfg.get("method", {}).get("name")

    if name in ("bpgs", "bpgs_alb"):
        from spectra.core.bpgs import BPGS

        m_cfg = cfg.get("method", {})
        return BPGS(
            num_tasks=len(cfg.tasks),
            s_min=m_cfg.get("s_min", -10.0),
            s_max=m_cfg.get("s_max", 10.0),
            s_init=m_cfg.get("s_init", 0.0),
        )

    cls = WEIGHTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown weighting method: {name}. Available: {list(WEIGHTER_REGISTRY.keys())}")

    params = cfg.get("params") or cfg.get("method", {}).get("params", {})
    return cls(num_tasks=len(cfg.tasks), **params)


__all__ = [
    "BaseWeighter",
    "StaticWeighter",
    "KendallWeighter",
    "UWSOWeighter",
    "PCGradWeighter",
    "GradNormProxyWeighter",
    "WEIGHTER_REGISTRY",
    "build_weighter",
]
