import torch
import pytest
from omegaconf import OmegaConf

from spectra.data.clinical.dataset import _is_outcome_positive_window
from spectra.data import datamodule as dm_mod
from spectra.data.clinical import dataset as clinical_ds


def test_outcome_policy_differs_from_phase_policy_on_resolving_window():
    # Obs has sepsis (shock), future has none => phase>0 but outcome should be 0
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32).numpy()
    assert _is_outcome_positive_window(labels, history_len=3) is False


class _DummyDs:
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "input": torch.zeros(2, 28),
            "targets": {"outcome": torch.tensor(0.0), "phase": torch.tensor(0)},
            "meta": {},
        }


def test_datamodule_forwards_sampler_target(monkeypatch):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "dataset": {
                "name": "clinical",
                "sepsis_boost": 10.0,
                "sampler_target": "outcome",
            },
            "train": {"batch_size": 2, "num_workers": 0},
        }
    )

    module = dm_mod.SPECTRADataModule(cfg)
    module.dataset_name = "clinical"
    module.train_ds = _DummyDs()

    seen = {}

    def _fake_sampler(**kwargs):
        seen.update(kwargs)
        return torch.utils.data.RandomSampler(module.train_ds)

    monkeypatch.setattr(dm_mod, "create_sepsis_aware_sampler", _fake_sampler)

    loader = module.train_dataloader()
    assert loader is not None
    assert seen["target"] == "outcome"


def test_sampler_rejects_unknown_target():
    class _Tiny:
        split = "train"
        subset_pct = 1.0
        root_path = __import__("pathlib").Path(".")

        def __len__(self):
            return 0

    with pytest.raises(ValueError):
        clinical_ds.create_sepsis_aware_sampler(_Tiny(), target="unknown")
