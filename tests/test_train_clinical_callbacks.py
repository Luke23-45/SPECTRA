from omegaconf import OmegaConf

from spectra.utils.callbacks import build_checkpoints, build_early_stopping


def _cfg(name="clinical"):
    return OmegaConf.create(
        {
            "dataset": {"name": name},
            "train": {"early_stop": True, "early_stop_patience": 3, "early_stop_min_delta": 1e-3},
        }
    )


def test_clinical_checkpoints_monitor_outcome_auc():
    ckpts = build_checkpoints(_cfg("clinical"), __import__("pathlib").Path("."))
    monitors = [c.monitor for c in ckpts]
    assert "val/outcome_AUC" in monitors
    assert "val/total_loss" in monitors


def test_clinical_early_stopping_uses_outcome_auc():
    es = build_early_stopping(_cfg("clinical"))
    assert es is not None
    assert es.monitor == "val/outcome_AUC"
    assert es.mode == "max"
