from omegaconf import OmegaConf

from spectra.baselines import build_weighter


def _cfg(method_name: str, method_inner_name: str, relative_loss_invariance: bool, mode: str):
    return OmegaConf.create(
        {
            "method_name": method_name,
            "method": {
                "name": method_inner_name,
                "s_min": -10.0,
                "s_max": 10.0,
                "s_init": 0.0,
                "relative_loss_invariance": relative_loss_invariance,
                "relative_loss_mode": mode,
            },
            "tasks": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        }
    )


def test_bpgs_baseline_stays_raw():
    cfg = _cfg("bpgs", "bpgs", False, "raw")
    w = build_weighter(cfg)
    assert w.relative_loss_invariance is False
    assert w.relative_loss_mode == "raw"


def test_bpgs_scaleinv_variant_enabled():
    cfg = _cfg("bpgs_scaleinv", "bpgs_scaleinv", True, "max")
    w = build_weighter(cfg)
    assert w.relative_loss_invariance is True
    assert w.relative_loss_mode == "max"
