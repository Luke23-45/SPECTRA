"""SPECTRA Core: B-PGS and ALB algorithmic modules."""

# Lazy imports to avoid circular dependencies and import crashes
# when only one module is needed.


def BPGSScaler(*args, **kwargs):
    from spectra.core.bpgs import BPGSScaler as _BPGSScaler
    return _BPGSScaler(*args, **kwargs)


def AsymmetricLatentBottleneck(*args, **kwargs):
    from spectra.core.alb import AsymmetricLatentBottleneck as _ALB
    return _ALB(*args, **kwargs)


__all__ = ["BPGSScaler", "AsymmetricLatentBottleneck"]
