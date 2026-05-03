"""
spectra/data/datamodule.py
--------------------------
Universal DataModule for SPECTRA experiments.

Orchestrates data acquisition, split management, and hardware-optimized 
DataLoader instantiation for heterogeneous MTL benchmarks.
"""

import logging
import os
import torch
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from omegaconf import DictConfig

from spectra.data.synthetic import SyntheticMTLDataset
from spectra.data.nyuv2.dataset import NYUv2Dataset, resolve_nyuv2_root
from spectra.data.rf1.dataset import RF1Dataset, resolve_rf1_root
from spectra.data.yeast.dataset import YeastDataset, resolve_yeast_root
from spectra.data.qm9.dataset import QM9Dataset, resolve_qm9_root
from spectra.data.clinical.dataset import ICUTrajectoryDataset, ICUSotaDataset, create_sepsis_aware_sampler, robust_collate_fn

logger = logging.getLogger("spectra.datamodule")


def _cfg_lookup(cfg: DictConfig, key: str, default=None):
    """Resolve top-level and nested dataset keys without dropping falsey values."""
    if key in cfg:
        return cfg.get(key)

    dataset_cfg = cfg.get("dataset", {})
    if key in dataset_cfg:
        return dataset_cfg.get(key)

    return default


def _loader_kwargs(dataset_name: str, num_workers: int, cfg: DictConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "pin_memory": True,
        "persistent_workers": (num_workers > 0),
    }

    if num_workers <= 0:
        return kwargs

    prefetch = cfg.train.get("prefetch_factor", None)
    if prefetch is None and dataset_name == "nyuv2":
        prefetch = 4
    if prefetch is None and dataset_name == "rf1":
        prefetch = 4
    if prefetch is None and dataset_name == "yeast":
        prefetch = 4
    if prefetch is None and dataset_name == "qm9":
        prefetch = 4
    if prefetch is not None:
        kwargs["prefetch_factor"] = int(prefetch)

    return kwargs


def _default_num_workers() -> int:
    """Choose a conservative default that avoids worker oversubscription."""
    cpu_count = os.cpu_count() or 1
    # Keep one core for the trainer / main process and avoid the prior hard-coded 4.
    return max(0, min(2, cpu_count - 1))


def _resolve_num_workers(cfg: DictConfig) -> int:
    configured = cfg.train.get("num_workers", None)
    if configured is None:
        return _default_num_workers()
    return int(configured)


def _use_nyuv2_batch_augmentation(cfg: DictConfig) -> bool:
    mode = _cfg_lookup(cfg, "batch_augmentation", "disabled")
    if mode == "disabled":
        return False
    if mode == "cpu":
        return True
    if mode == "cuda":
        return torch.cuda.is_available()
    raise ValueError(f"Unsupported NYUv2 batch_augmentation mode: {mode}")


class SPECTRADataModule(pl.LightningDataModule):
    """
    Central dispatcher for all SPECTRA benchmarks.
    
    NASA-Grade guarantees:
    - Rank-aware seeding for deterministic DDP streams.
    - Tiered Acquisition: Automatic fallback from Cloud to Local Build.
    - Schema Integrity: Cross-dataset standardization via {input, targets, meta}.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Robustly identify dataset name from either global or nested config
        self.dataset_name = cfg.get("dataset_name") or cfg.get("dataset", {}).get("name", "synthetic")
        
        # Placeholders
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        """Tiered Acquisition logic (Rank 0 only)."""
        if self.dataset_name == "clinical":
            from spectra.data.clinical.dataset import ensure_data_ready
            # Robust extraction of clinical parameters
            dataset_dir = self.cfg.get("dataset_dir") or self.cfg.get("dataset", {}).get("dataset_dir", "data/ready")
            hf_repo = self.cfg.get("hf_repo") or self.cfg.get("dataset", {}).get("hf_repo", None)
            force_download = self.cfg.get("force_download") or self.cfg.get("dataset", {}).get("force_download", False)
            
            ensure_data_ready(
                dataset_dir=dataset_dir,
                hf_repo_id=hf_repo,
                force_download=force_download
            )
        elif self.dataset_name == "nyuv2":
            # Validate that LMDB data exists before training starts
            root = _cfg_lookup(self.cfg, "root", "datasets/nyuv2_lmdb")
            from pathlib import Path
            root_path = resolve_nyuv2_root(root)
            for split in ["train", "val"]:
                lmdb_path = root_path / split / "data.lmdb"
                index_path = root_path / f"{split}_index.json"
                if not lmdb_path.exists() or not index_path.exists():
                    logger.warning(
                        f"[NYUv2] Missing data for split '{split}'. "
                        f"Run: python -m spectra.data.nyuv2.nyuv2_lmdb_sota"
                    )
        elif self.dataset_name == "rf1":
            root = _cfg_lookup(self.cfg, "root", "datasets/rf1")
            root_path = resolve_rf1_root(root)
            for required_path in [
                root_path / "metadata.json",
                root_path / "train" / "data.npz",
                root_path / "val" / "data.npz",
            ]:
                if not required_path.exists():
                    logger.warning(
                        "[RF1] Missing prepared data at '%s'. "
                        "Run: python -m spectra.data.rf1.ingest or python scripts/data/rf1_generate.py",
                        required_path,
                    )
        elif self.dataset_name == "yeast":
            root = _cfg_lookup(self.cfg, "root", "datasets/yeast")
            root_path = resolve_yeast_root(root)
            for required_path in [
                root_path / "metadata.json",
                root_path / "train" / "data.npz",
                root_path / "val" / "data.npz",
            ]:
                if not required_path.exists():
                    logger.warning(
                        "[YEAST] Missing prepared data at '%s'. "
                        "Run: python -m spectra.data.yeast.ingest or python scripts/data/yeast_generate.py",
                        required_path,
                    )
        elif self.dataset_name == "qm9":
            root = _cfg_lookup(self.cfg, "root", "datasets/qm9")
            root_path = resolve_qm9_root(root)
            for required_path in [
                root_path / "metadata.json",
                root_path / "train" / "data.npz",
                root_path / "val" / "data.npz",
            ]:
                if not required_path.exists():
                    logger.warning(
                        "[QM9] Missing prepared data at '%s'. "
                        "Run: python -m spectra.data.qm9.ingest or python scripts/data/qm9_generate.py",
                        required_path,
                    )

    def setup(self, stage: Optional[str] = None):
        """Instantiate datasets across all DDP ranks."""
        if self.dataset_name == "synthetic":
            self.train_ds = SyntheticMTLDataset(
                n_samples=self.cfg.data.n_train,
                input_dim=self.cfg.model.input_dim,
                hidden_dim=self.cfg.model.d_model,
                seed=self.cfg.seed,
                mapping_seed=self.cfg.seed
            )
            self.val_ds = SyntheticMTLDataset(
                n_samples=self.cfg.data.n_val,
                input_dim=self.cfg.model.input_dim,
                hidden_dim=self.cfg.model.d_model,
                seed=self.cfg.seed + 1,
                mapping_seed=self.cfg.seed  # Critical: same function/mapping as train
            )

        elif self.dataset_name == "nyuv2":
            root = _cfg_lookup(self.cfg, "root")
            subset_pct = _cfg_lookup(self.cfg, "subset_pct", 1.0)
            subset_seed = _cfg_lookup(self.cfg, "subset_seed", 42)
            train_subset_file = _cfg_lookup(self.cfg, "train_subset_file", None)
            train_subset_id = _cfg_lookup(self.cfg, "train_subset_id", None)
            normalize_rgb = _cfg_lookup(self.cfg, "normalize_rgb", False)
            augmentation = _cfg_lookup(self.cfg, "augmentation", True)
            use_batch_aug = _use_nyuv2_batch_augmentation(self.cfg)
            
            self.train_ds = NYUv2Dataset(
                root=root,
                split="train",
                augmentation=(augmentation and not use_batch_aug),
                subset_pct=subset_pct,
                subset_seed=subset_seed,
                subset_file=train_subset_file,
                subset_id=train_subset_id,
                normalize_rgb=(normalize_rgb and not use_batch_aug),
            )
            self.val_ds = NYUv2Dataset(
                root=root,
                split="val",
                augmentation=False,
                subset_pct=1.0,
                subset_seed=subset_seed,
                normalize_rgb=normalize_rgb,
            )

        elif self.dataset_name == "rf1":
            root = _cfg_lookup(self.cfg, "root")
            subset_pct = _cfg_lookup(self.cfg, "subset_pct", 1.0)
            subset_seed = _cfg_lookup(self.cfg, "subset_seed", 42)
            normalize_inputs = _cfg_lookup(self.cfg, "normalize_inputs", True)

            self.train_ds = RF1Dataset(
                root=root,
                split="train",
                normalize_inputs=normalize_inputs,
                subset_pct=subset_pct,
                subset_seed=subset_seed,
            )
            self.val_ds = RF1Dataset(
                root=root,
                split="val",
                normalize_inputs=normalize_inputs,
                subset_pct=subset_pct,
                subset_seed=subset_seed,
            )

        elif self.dataset_name == "yeast":
            root = _cfg_lookup(self.cfg, "root")
            subset_pct = _cfg_lookup(self.cfg, "subset_pct", 1.0)
            subset_seed = _cfg_lookup(self.cfg, "subset_seed", 42)
            normalize_inputs = _cfg_lookup(self.cfg, "normalize_inputs", True)

            self.train_ds = YeastDataset(
                root=root,
                split="train",
                normalize_inputs=normalize_inputs,
                subset_pct=subset_pct,
                subset_seed=subset_seed,
            )
            self.val_ds = YeastDataset(
                root=root,
                split="val",
                normalize_inputs=normalize_inputs,
                subset_pct=subset_pct,
                subset_seed=subset_seed,
            )

        elif self.dataset_name == "qm9":
            root = _cfg_lookup(self.cfg, "root")
            subset_pct = _cfg_lookup(self.cfg, "subset_pct", 1.0)
            subset_seed = _cfg_lookup(self.cfg, "subset_seed", 42)
            normalize_inputs = _cfg_lookup(self.cfg, "normalize_inputs", True)

            self.train_ds = QM9Dataset(
                root=root,
                split="train",
                normalize_inputs=normalize_inputs,
                subset_pct=subset_pct,
                subset_seed=subset_seed,
            )
            self.val_ds = QM9Dataset(
                root=root,
                split="val",
                normalize_inputs=normalize_inputs,
                subset_pct=subset_pct,
                subset_seed=subset_seed,
            )

        elif self.dataset_name == "clinical":
            dataset_dir = self.cfg.get("dataset_dir") or self.cfg.get("dataset", {}).get("dataset_dir", "data/ready")
            subset_pct = self.cfg.get("subset_pct") or self.cfg.get("dataset", {}).get("subset_pct", 1.0)
            
            self.train_ds = ICUSotaDataset(
                dataset_dir=dataset_dir,
                split="train",
                augment_noise=self.cfg.train.get("augment_noise", 0.0),
                augment_mask_prob=self.cfg.train.get("augment_mask_prob", 0.0),
                subset_pct=subset_pct
            )
            self.val_ds = ICUTrajectoryDataset(
                dataset_dir=dataset_dir,
                split="val",
                subset_pct=subset_pct
            )

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        logger.info(f"[DataModule] Setup complete: {self.dataset_name} (Train={len(self.train_ds)}, Val={len(self.val_ds)})")

    def train_dataloader(self):
        batch_size = self.cfg.train.batch_size
        num_workers = _resolve_num_workers(self.cfg)
        loader_kwargs = _loader_kwargs(self.dataset_name, num_workers, self.cfg)
        
        # Clinical requires specialized weighted sampler for sepsis oversampling
        if self.dataset_name == "clinical":
            sepsis_boost = self.cfg.get("sepsis_boost") or self.cfg.get("dataset", {}).get("sepsis_boost", 5.0)
            sampler_target = self.cfg.get("sampler_target") or self.cfg.get("dataset", {}).get("sampler_target", "outcome")
            sampler_max_samples = (
                self.cfg.train.get("sampler_max_samples", None)
                or self.cfg.get("sampler_max_samples")
                or self.cfg.get("dataset", {}).get("sampler_max_samples", None)
            )
            
            sampler = create_sepsis_aware_sampler(
                dataset=self.train_ds,
                sepsis_boost_factor=sepsis_boost,
                max_samples=sampler_max_samples if sampler_max_samples is not None else 100000,
                seed=self.cfg.seed,
                target=sampler_target,
            )
            # Sampler is inherently Distributed-aware (Axe v3.2 StatefulSampler)
            return DataLoader(
                self.train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=robust_collate_fn,
                drop_last=True,
                **loader_kwargs,
            )
        
        # Others (Vision/Synthetic)
        sampler = None
            
        collate_fn = None
        if self.dataset_name == "nyuv2":
            collate_fn = NYUv2Dataset.collate_fn
        elif self.dataset_name == "rf1":
            collate_fn = RF1Dataset.collate_fn
        elif self.dataset_name == "yeast":
            collate_fn = YeastDataset.collate_fn
        elif self.dataset_name == "qm9":
            collate_fn = QM9Dataset.collate_fn
        elif self.dataset_name == "synthetic":
            collate_fn = SyntheticMTLDataset.collate_fn

        return DataLoader(
            self.train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            **loader_kwargs,
        )

    def val_dataloader(self):
        num_workers = _resolve_num_workers(self.cfg)
        loader_kwargs = _loader_kwargs(self.dataset_name, num_workers, self.cfg)
        collate_fn = None
        if self.dataset_name == "nyuv2":
            collate_fn = NYUv2Dataset.collate_fn
        elif self.dataset_name == "rf1":
            collate_fn = RF1Dataset.collate_fn
        elif self.dataset_name == "yeast":
            collate_fn = YeastDataset.collate_fn
        elif self.dataset_name == "qm9":
            collate_fn = QM9Dataset.collate_fn
        elif self.dataset_name == "clinical":
            collate_fn = robust_collate_fn
        elif self.dataset_name == "synthetic":
            collate_fn = SyntheticMTLDataset.collate_fn

        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
