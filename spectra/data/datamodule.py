"""
spectra/data/datamodule.py
--------------------------
Universal DataModule for SPECTRA experiments.

Orchestrates data acquisition, split management, and hardware-optimized 
DataLoader instantiation for heterogeneous MTL benchmarks.
"""

import logging
import torch
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from omegaconf import DictConfig

from spectra.data.synthetic import SyntheticMTLDataset
from spectra.data.nyuv2.dataset import NYUv2Dataset
from spectra.data.clinical.dataset import ICUTrajectoryDataset, ICUSotaDataset, create_sepsis_aware_sampler, robust_collate_fn

logger = logging.getLogger("spectra.datamodule")


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
        self.dataset_name = cfg.dataset.get("name", "synthetic")
        
        # Placeholders
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        """Tiered Acquisition logic (Rank 0 only)."""
        if self.dataset_name == "clinical":
            from spectra.data.clinical.dataset import ensure_data_ready
            ensure_data_ready(
                dataset_dir=self.cfg.dataset.get("dataset_dir", "data/ready"),
                hf_repo_id=self.cfg.dataset.get("hf_repo", None),
                force_download=self.cfg.dataset.get("force_download", False)
            )
        elif self.dataset_name == "nyuv2":
            # NYUv2 usually assumes local extraction from MTAN binaries
            pass

    def setup(self, stage: Optional[str] = None):
        """Instantiate datasets across all DDP ranks."""
        if self.dataset_name == "synthetic":
            self.train_ds = SyntheticMTLDataset(
                n_samples=self.cfg.data.n_train,
                input_dim=self.cfg.model.input_dim,
                hidden_dim=self.cfg.model.d_model,
                seed=self.cfg.seed
            )
            self.val_ds = SyntheticMTLDataset(
                n_samples=self.cfg.data.n_val,
                input_dim=self.cfg.model.input_dim,
                hidden_dim=self.cfg.model.d_model,
                seed=self.cfg.seed + 1
            )
            
        elif self.dataset_name == "nyuv2":
            self.train_ds = NYUv2Dataset(
                root=self.cfg.dataset.root,
                split="train",
                augmentation=True,
                subset_pct=self.cfg.dataset.get("subset_pct", 1.0)
            )
            self.val_ds = NYUv2Dataset(
                root=self.cfg.dataset.root,
                split="val",
                augmentation=False
            )

        elif self.dataset_name == "clinical":
            dataset_dir = self.cfg.dataset.get("dataset_dir", "data/ready")
            subset_pct = self.cfg.dataset.get("subset_pct", 1.0)
            
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
        num_workers = self.cfg.train.get("num_workers", 4)
        
        # Clinical requires specialized weighted sampler for sepsis oversampling
        if self.dataset_name == "clinical":
            sampler = create_sepsis_aware_sampler(
                dataset=self.train_ds,
                sepsis_boost_factor=self.cfg.dataset.get("sepsis_boost", 5.0),
                seed=self.cfg.seed
            )
            # Sampler is inherently Distributed-aware (Axe v3.2 StatefulSampler)
            return DataLoader(
                self.train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=robust_collate_fn,
                pin_memory=True,
                drop_last=True
            )
        
        # Others (Vision/Synthetic)
        sampler = None
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(self.train_ds, shuffle=True, seed=self.cfg.seed)
            
        collate_fn = None
        if self.dataset_name == "nyuv2":
            collate_fn = NYUv2Dataset.collate_fn
        elif self.dataset_name == "synthetic":
            collate_fn = SyntheticMTLDataset.collate_fn

        return DataLoader(
            self.train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        collate_fn = None
        if self.dataset_name == "nyuv2":
            collate_fn = NYUv2Dataset.collate_fn
        elif self.dataset_name == "clinical":
            collate_fn = robust_collate_fn
        elif self.dataset_name == "synthetic":
            collate_fn = SyntheticMTLDataset.collate_fn

        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True
        )
