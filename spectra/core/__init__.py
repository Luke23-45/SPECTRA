"""SPECTRA Core: B-PGS and ALB algorithmic modules."""

# Lazy imports to avoid circular dependencies and import crashes
# when only one module is needed.


def BPGS(*args, **kwargs):
    from spectra.core.bpgs import BPGS as _BPGS
    return _BPGS(*args, **kwargs)


def AsymmetricLatentBottleneck(*args, **kwargs):
    from spectra.core.alb import AsymmetricLatentBottleneck as _ALB
    return _ALB(*args, **kwargs)


__all__ = ["BPGS", "AsymmetricLatentBottleneck"]
