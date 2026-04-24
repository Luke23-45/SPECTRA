from pathlib import Path

import yaml

from omegaconf import OmegaConf

from spectra.utils.callbacks import build_early_stopping


def test_gradnorm_proxy_default_has_no_ema_smoothing():
    cfg_path = Path("configs/method/gradnorm_proxy.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    assert cfg["method"]["params"]["ema_decay"] == 0.0


def test_nyuv2_standard_budget_is_capped_and_early_stopped():
    cfg_path = Path("configs/dataset/nyuv2.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    assert cfg["train"]["epochs"] == 100
    assert cfg["train"]["early_stop"] is True
    assert cfg["train"]["early_stop_patience"] == 15


def test_nyuv2_early_stopping_monitors_miou():
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "nyuv2"},
            "train": {
                "early_stop": True,
                "early_stop_patience": 15,
                "early_stop_min_delta": 1e-4,
            },
        }
    )

    es = build_early_stopping(cfg)
    assert es is not None
    assert es.monitor == "val/miou"
    assert es.mode == "max"
