from pathlib import Path

import torch
from omegaconf import OmegaConf

from spectra.architectures.builder import build_model
from spectra.modules.vision import VisionSPECTRAModule
from spectra.engine.optimizers.standard import StandardEngine
from spectra.utils.callbacks import build_checkpoints, build_early_stopping


def test_nyuv2_callbacks_use_top_level_dataset_name():
    cfg = OmegaConf.create(
        {
            "dataset_name": "nyuv2",
            "train": {"save_ckpt": True, "early_stop": True, "early_stop_patience": 3},
        }
    )

    ckpts = build_checkpoints(cfg, Path("."))
    monitors = [c.monitor for c in ckpts]

    assert "val/miou" in monitors
    assert "val/total_loss" in monitors

    es = build_early_stopping(cfg)
    assert es is not None
    assert es.monitor == "val/miou"
    assert es.mode == "max"


def test_dense_alb_both_manifold_concatenates_channels_not_width():
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": "segnet",
                "input_channels": 3,
                "d_model": 16,
            },
            "use_alb": True,
            "method": {
                "use_alb": True,
                "n_expert_layers": 1,
                "expert_init": "orthogonal",
            },
            "tasks": [
                {
                    "name": "segmentation",
                    "type": "dense_classification",
                    "num_classes": 13,
                    "manifold": "both",
                }
            ],
        }
    )

    model = build_model(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)

    assert out["segmentation"].shape == (2, 13, 32, 32)


def test_vision_module_initializes_with_dense_metrics():
    cfg = OmegaConf.create(
        {
            "method_name": "static",
            "model": {"backbone": "segnet", "input_channels": 3, "d_model": 16},
            "method": {"name": "static"},
            "tasks": [
                {
                    "name": "segmentation",
                    "type": "dense_classification",
                    "loss": "cross_entropy",
                    "num_classes": 13,
                    "ignore_index": 255,
                },
                {"name": "depth", "type": "dense_regression", "loss": "masked_l1", "output_dim": 1},
                {"name": "normals", "type": "dense_regression", "loss": "cosine_dense", "output_dim": 3},
            ],
            "train": {"lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 1, "min_lr": 1e-5},
        }
    )

    module = VisionSPECTRAModule(cfg, StandardEngine())

    assert set(module._val_metrics.keys()) == {"segmentation", "depth", "normals"}
