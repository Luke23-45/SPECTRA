from pathlib import Path

import torch
import pytest
from omegaconf import OmegaConf

from spectra.architectures.builder import build_model
from spectra.data.datamodule import SPECTRADataModule
from spectra.data.nyuv2.dataset import resolve_nyuv2_root
from spectra.modules.vision import VisionSPECTRAModule
from spectra.engine.optimizers.standard import StandardEngine
from spectra.data.nyuv2.transforms import NYUv2TestTransform
from spectra.train.preflight import preflight_check
from spectra.train.artifacts import (
    resolve_resume_checkpoint,
    save_experiment_config,
    save_experiment_metadata,
    generate_experiment_metadata
)
from spectra.utils.callbacks import build_checkpoints, build_early_stopping


def test_nyuv2_callbacks_use_top_level_dataset_name():
    cfg = OmegaConf.create(
        {
            "dataset_name": "nyuv2",
            "train": {"save_ckpt": True, "early_stop": True, "early_stop_patience": 3, "checkpoint_every_minutes": 10},
        }
    )

    ckpts = build_checkpoints(cfg, Path("."))
    monitors = [c.monitor for c in ckpts]

    assert "val/miou" in monitors
    assert "val/total_loss" in monitors
    assert any(getattr(c, "_train_time_interval", None) is not None for c in ckpts)

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


def test_nyuv2_datamodule_respects_configured_augmentation_and_subset_seed(tmp_path):
    import json
    import lmdb
    import numpy as np

    root = tmp_path / "nyuv2"
    height, width = 32, 32

    for split, count in [("train", 5), ("val", 3)]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(split_dir / "data.lmdb"), map_size=10**7, subdir=False)
        episodes = []
        with env.begin(write=True) as txn:
            for i in range(count):
                image = (np.random.rand(height, width, 3) * 255.0).astype(np.uint8)
                label = np.random.randint(0, 13, size=(height, width), dtype=np.uint8)
                depth = (np.random.rand(height, width, 1) * 5.0).astype(np.float16)
                normal = np.random.randn(height, width, 3).astype(np.float32)
                norm = np.linalg.norm(normal, axis=-1, keepdims=True)
                normal = (normal / np.maximum(norm, 1e-6)).astype(np.float16)
                keys = {
                    "image_key": f"img_{i}",
                    "label_key": f"lbl_{i}",
                    "depth_key": f"dep_{i}",
                    "normal_key": f"nrm_{i}",
                }
                txn.put(keys["image_key"].encode("ascii"), image.tobytes())
                txn.put(keys["label_key"].encode("ascii"), label.tobytes())
                txn.put(keys["depth_key"].encode("ascii"), depth.tobytes())
                txn.put(keys["normal_key"].encode("ascii"), normal.tobytes())
                episodes.append({"shape_hw": [height, width], **keys})
        env.close()

        with open(root / f"{split}_index.json", "w", encoding="utf-8") as handle:
            json.dump({"episodes": episodes, "metadata": {}}, handle)

    cfg = OmegaConf.create(
        {
            "dataset_name": "nyuv2",
            "root": str(root),
            "augmentation": False,
            "subset_pct": 0.4,
            "subset_seed": 7,
            "normalize_rgb": False,
            "train": {"batch_size": 2, "num_workers": 0},
        }
    )

    dm = SPECTRADataModule(cfg)
    dm.setup()

    assert len(dm.train_ds) == 2
    assert dm.train_ds.indices == [1, 2]
    assert isinstance(dm.train_ds.transform, NYUv2TestTransform)


def test_nyuv2_preflight_requires_index_files(tmp_path):
    root = tmp_path / "nyuv2"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir(parents=True)
    (root / "train" / "data.lmdb").write_bytes(b"")
    (root / "val" / "data.lmdb").write_bytes(b"")

    cfg = OmegaConf.create(
        {
            "dataset_name": "nyuv2",
            "root": str(root),
            "method_name": "static",
            "tasks": [{"name": "segmentation", "loss": "cross_entropy"}],
            "train": {"batch_size": 2},
        }
    )

    with pytest.raises(SystemExit) as excinfo:
        preflight_check(cfg, Path("."))

    message = str(excinfo.value)
    assert "Pre-flight check FAILED" in message




def test_nyuv2_preflight_accepts_bpgs_scaleinv_method(tmp_path):
    root = tmp_path / "nyuv2"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir(parents=True)
    (root / "train" / "data.lmdb").write_bytes(b"")
    (root / "val" / "data.lmdb").write_bytes(b"")
    (root / "train_index.json").write_text("{\"episodes\": [], \"metadata\": {}}", encoding="utf-8")
    (root / "val_index.json").write_text("{\"episodes\": [], \"metadata\": {}}", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "dataset_name": "nyuv2",
            "root": str(root),
            "method_name": "bpgs_scaleinv",
            "tasks": [{"name": "segmentation", "loss": "cross_entropy"}],
            "train": {"batch_size": 2},
        }
    )

    preflight_check(cfg, Path("."))


def test_nyuv2_root_resolution_supports_nested_data_dir(tmp_path):
    nested = tmp_path / "nyuv2_lmdb" / "data"
    (nested / "train").mkdir(parents=True)
    (nested / "val").mkdir(parents=True)

    resolved = resolve_nyuv2_root(tmp_path / "nyuv2_lmdb")

    assert resolved == nested


def test_resume_from_auto_uses_last_checkpoint_in_stable_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "outputs" / "pcgrad_nyuv2_s42"
    ckpt_path = artifact_dir / "checkpoints" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint")

    cfg = OmegaConf.create({"resume_from": "auto"})

    resolved = resolve_resume_checkpoint(cfg, artifact_dir)

    assert resolved == ckpt_path


def test_preflight_uses_stable_artifact_dir_for_resume_auto(tmp_path):
    artifact_dir = tmp_path / "outputs" / "pcgrad_nyuv2_s42"
    ckpt_path = artifact_dir / "checkpoints" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint")

    root = tmp_path / "nyuv2"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir(parents=True)
    (root / "train" / "data.lmdb").write_bytes(b"")
    (root / "val" / "data.lmdb").write_bytes(b"")
    (root / "train_index.json").write_text("{\"episodes\": [], \"metadata\": {}}", encoding="utf-8")
    (root / "val_index.json").write_text("{\"episodes\": [], \"metadata\": {}}", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "dataset_name": "nyuv2",
            "root": str(root),
            "method_name": "static",
            "run_name": "pcgrad_nyuv2_s42",
            "output_dir": str(artifact_dir),
            "resume_from": "auto",
            "tasks": [{"name": "segmentation", "loss": "cross_entropy"}],
            "train": {"batch_size": 2},
        }
    )

    preflight_check(cfg, tmp_path / "hydra-runtime-shell")


def test_save_experiment_config_saves_resolved_yaml(tmp_path):
    """Test that config is saved to artifact_dir/config.yaml."""
    cfg = OmegaConf.create({
        "dataset_name": "nyuv2",
        "method_name": "pcgrad",
        "run_name": "${method_name}_${dataset_name}_s${seed}",
        "seed": 42,
        "train": {"epochs": 100, "batch_size": 2}
    })
    
    artifact_dir = tmp_path / "outputs" / "test_run"
    artifact_dir.mkdir(parents=True)
    
    config_path = save_experiment_config(cfg, artifact_dir)
    
    assert config_path == artifact_dir / "config.yaml"
    assert config_path.exists()
    
    # Verify the saved config can be loaded
    loaded_cfg = OmegaConf.load(config_path)
    assert loaded_cfg.dataset_name == "nyuv2"
    assert loaded_cfg.method_name == "pcgrad"
    assert loaded_cfg.run_name == "pcgrad_nyuv2_s42"
    assert loaded_cfg.train.epochs == 100


def test_save_experiment_metadata_generates_valid_json(tmp_path):
    """Test that metadata.json is generated with all required fields."""
    cfg = OmegaConf.create({
        "dataset_name": "nyuv2",
        "method_name": "pcgrad",
        "run_name": "pcgrad_nyuv2_s42",
        "seed": 42,
        "num_classes": 13,
        "ignore_index": 255,
        "image_height": 288,
        "image_width": 384,
        "train_size": 795,
        "val_size": 654,
        "augmentation": True,
        "subset_pct": 1.0,
        "subset_seed": 42,
        "train": {
            "epochs": 200,
            "batch_size": 2,
            "lr": 0.0005,
            "min_lr": 0.000001,
            "warmup_steps": 100,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "precision": "16-mixed",
            "num_workers": 2,
            "deterministic": False,
            "early_stop": True,
            "early_stop_patience": 10,
            "checkpoint_every_minutes": 10
        },
        "tasks": [
            {"name": "segmentation", "type": "dense_classification", "loss": "cross_entropy", "manifold": "planner", "weight": 1.0},
            {"name": "depth", "type": "dense_regression", "loss": "masked_l1", "manifold": "planner", "weight": 1.0},
            {"name": "normals", "type": "dense_regression", "loss": "cosine_dense", "manifold": "expert", "weight": 1.0}
        ]
    })
    
    artifact_dir = tmp_path / "outputs" / "test_run"
    artifact_dir.mkdir(parents=True)
    
    metadata_path = save_experiment_metadata(cfg, artifact_dir)
    
    assert metadata_path == artifact_dir / "metadata.json"
    assert metadata_path.exists()
    
    # Verify the saved metadata can be loaded and has required fields
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Check top-level sections
    assert "experiment" in metadata
    assert "system" in metadata
    assert "git" in metadata
    assert "dataset" in metadata
    assert "training" in metadata
    assert "tasks" in metadata
    assert "paths" in metadata
    assert "timestamp" in metadata
    
    # Check experiment section
    assert metadata["experiment"]["run_name"] == "pcgrad_nyuv2_s42"
    assert metadata["experiment"]["method"] == "pcgrad"
    assert metadata["experiment"]["dataset"] == "nyuv2"
    assert metadata["experiment"]["seed"] == 42
    
    # Check dataset section for NYUv2-specific fields
    assert metadata["dataset"]["name"] == "nyuv2"
    assert metadata["dataset"]["num_classes"] == 13
    assert metadata["dataset"]["image_height"] == 288
    assert metadata["dataset"]["image_width"] == 384
    assert metadata["dataset"]["train_size"] == 795
    assert metadata["dataset"]["val_size"] == 654
    
    # Check training section
    assert metadata["training"]["epochs"] == 200
    assert metadata["training"]["batch_size"] == 2
    assert metadata["training"]["lr"] == 0.0005
    assert metadata["training"]["precision"] == "16-mixed"
    assert metadata["training"]["checkpoint_every_minutes"] == 10

    # Check resume section
    assert metadata["resume"]["requested"] is None
    assert metadata["resume"]["resolved_checkpoint"] is None
    assert metadata["resume"]["auto_resume_found"] is None
    
    # Check tasks section
    assert len(metadata["tasks"]) == 3
    assert metadata["tasks"][0]["name"] == "segmentation"
    assert metadata["tasks"][1]["name"] == "depth"
    assert metadata["tasks"][2]["name"] == "normals"
    
    # Check system section
    assert "torch_version" in metadata["system"]
    assert "cuda_available" in metadata["system"]
    assert "python_version" in metadata["system"]
    
    # Check git section (may be unknown if not in git repo)
    assert "commit_hash" in metadata["git"]
    assert "branch" in metadata["git"]
    assert "is_dirty" in metadata["git"]


def test_generate_experiment_metadata_handles_missing_dataset_specifics(tmp_path):
    """Test that metadata generation works for non-NYUv2 datasets."""
    cfg = OmegaConf.create({
        "dataset_name": "synthetic",
        "method_name": "static",
        "run_name": "static_synthetic_s42",
        "seed": 42,
        "augmentation": False,
        "subset_pct": 1.0,
        "subset_seed": 42,
        "train": {
            "epochs": 50,
            "batch_size": 32,
            "lr": 0.001,
            "min_lr": 0.00001,
            "warmup_steps": 50,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "precision": "32",
            "num_workers": 4,
            "deterministic": False,
            "early_stop": False,
            "early_stop_patience": 0,
            "checkpoint_every_minutes": 0
        },
        "tasks": [
            {"name": "task1", "type": "classification", "loss": "mse", "manifold": "planner", "weight": 1.0}
        ]
    })
    
    artifact_dir = tmp_path / "outputs" / "test_run"
    artifact_dir.mkdir(parents=True)
    
    metadata = generate_experiment_metadata(cfg, artifact_dir)
    
    # Should not have NYUv2-specific fields
    assert "num_classes" not in metadata["dataset"]
    assert "image_height" not in metadata["dataset"]
    assert "image_width" not in metadata["dataset"]
    
    # Should have generic dataset fields
    assert metadata["dataset"]["name"] == "synthetic"
    assert metadata["dataset"]["augmentation"] is False
    assert metadata["dataset"]["subset_pct"] == 1.0


def test_generate_experiment_metadata_records_auto_resume_status(tmp_path):
    artifact_dir = tmp_path / "outputs" / "pcgrad_nyuv2_s42"
    ckpt_path = artifact_dir / "checkpoints" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint")

    cfg = OmegaConf.create(
        {
            "dataset_name": "nyuv2",
            "method_name": "pcgrad",
            "run_name": "pcgrad_nyuv2_s42",
            "seed": 42,
            "resume_from": "auto",
            "train": {"epochs": 1, "batch_size": 2},
            "tasks": [],
        }
    )

    metadata = generate_experiment_metadata(cfg, artifact_dir)

    assert metadata["resume"]["requested"] == "auto"
    assert metadata["resume"]["resolved_checkpoint"] == str(ckpt_path)
    assert metadata["resume"]["auto_resume_found"] is True
