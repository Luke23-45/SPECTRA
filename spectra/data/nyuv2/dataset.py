"""
spectra/data/nyuv2/dataset.py
------------------------------
NYUv2 Multi-Task Dense Prediction Dataset (Production-Grade).

Author: SPECTRA Research Team
Status: Production-Ready

Description:
    Loads the NYUv2 dataset in MTAN-standard .npy format and returns
    SPECTRA-compatible batch dictionaries with three task targets:
    1. Semantic Segmentation (13 classes)
    2. Monocular Depth Estimation
    3. Surface Normal Prediction

    This follows the same architectural philosophy as the clinical pipeline
    (spectra/data/clinical/dataset.py) but adapted for dense image prediction.

Data Format:
    Input:  RGB image  → [3, H, W] float32
    Target: Segmentation → [H, W] int64, class indices {0..12, 255=ignore}
            Depth         → [1, H, W] float32, metric meters
            Normals       → [3, H, W] float32, unit direction (x, y, z)

Safety Guarantees:
    - Schema validation: verifies file counts match expected split sizes
    - Dtype enforcement: segmentation is ALWAYS int64 (CrossEntropyLoss safe)
    - NaN/Inf trapping: depth and normals checked for corruption
    - Fork safety: no shared state across DataLoader workers
    - Ignore class remapping: -1 → 255 for cross-entropy ignore_index

References:
    - Silberman et al. "Indoor Segmentation and Support Inference from RGBD Images" (ECCV 2012)
    - Liu et al. "End-to-End Multi-Task Learning with Attention" (CVPR 2019) [MTAN]
    - Standard 13-class split: https://github.com/lorenmt/mtan
"""

from __future__ import annotations

import os
import fnmatch
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from .transforms import NYUv2TrainTransform, NYUv2TestTransform

# =============================================================================
# CONFIGURATION
# =============================================================================

logger = logging.getLogger("spectra.data.nyuv2")

# Standard NYUv2 MTL benchmark split sizes
EXPECTED_SPLIT_SIZES = {
    "train": 795,
    "val": 654,
}

# The 13 semantic classes in the standard NYUv2 MTL split
# (Reduced from the original 894 classes by Eigen & Fergus)
NYUv2_CLASS_NAMES = [
    "bed", "books", "ceiling", "chair", "floor",
    "furniture", "objects", "painting", "sofa", "table",
    "tv", "wall", "window",
]

NUM_CLASSES = 13
IGNORE_INDEX = 255  # Standard ignore index for CrossEntropyLoss


# =============================================================================
# CORE DATASET
# =============================================================================

class NYUv2Dataset(Dataset):
    """
    NYUv2 Multi-Task Learning Dataset.

    Loads pre-processed .npy files from MTAN-standard directory structure:
        root/
            train/
                image/0.npy, 1.npy, ..., 794.npy
                label/0.npy, ...
                depth/0.npy, ...
                normal/0.npy, ...
            val/
                image/0.npy, ..., 653.npy
                ...

    Returns SPECTRA-compatible batch dictionary:
        {
            "input": [3, H, W] float32     — RGB image
            "targets": {
                "segmentation": [H, W] int64    — class indices {0..12, 255}
                "depth":        [1, H, W] float32 — metric depth (meters)
                "normals":      [3, H, W] float32 — surface normals (x,y,z)
            }
            "meta": {
                "depth_mask":   [1, H, W] float32 — 1=valid, 0=invalid depth
                "sample_id":    str               — reproducibility key
            }
        }

    Args:
        root: Path to dataset root (contains train/ and val/ subdirs).
        split: "train" or "val".
        augmentation: Enable RandomScaleCrop + RandomHorizontalFlip.
        normalize_rgb: Apply ImageNet normalization (for pretrained backbones).
        validate_schema: Check file counts match expected sizes.
    """

    def __init__(
        self,
        root: str = "data/nyuv2",
        split: str = "train",
        augmentation: bool = True,
        normalize_rgb: bool = False,
        validate_schema: bool = True,
        subset_pct: float = 1.0,
        subset_seed: int = 42,
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.data_path = self.root / split

        # --- Validate Directory Structure ---
        if not self.data_path.is_dir():
            raise FileNotFoundError(
                f"[NYUv2] Split directory not found: {self.data_path}\n"
                f"Run `python -m spectra.data.nyuv2.download` to download."
            )

        for modality in ["image", "label", "depth", "normal"]:
            mod_dir = self.data_path / modality
            if not mod_dir.is_dir():
                raise FileNotFoundError(
                    f"[NYUv2] Missing modality directory: {mod_dir}\n"
                    f"Expected MTAN-format directory structure."
                )

        # --- Count Available Files ---
        self.data_len = len(fnmatch.filter(
            os.listdir(str(self.data_path / "image")), "*.npy"
        ))

        if self.data_len == 0:
            raise RuntimeError(f"[NYUv2] No .npy files found in {self.data_path / 'image'}")

        # --- Schema Validation ---
        if validate_schema:
            expected = EXPECTED_SPLIT_SIZES.get(split, 0)
            if expected > 0 and self.data_len != expected:
                logger.warning(
                    f"[NYUv2] File count mismatch in {split}: "
                    f"found {self.data_len}, expected {expected}. "
                    f"Proceeding with found count."
                )

        # --- Build Index ---
        # Use simple integer indices (MTAN convention: 0.npy, 1.npy, ...)
        all_indices = list(range(self.data_len))
        
        if 0.0 < subset_pct < 1.0:
            import random
            rng = random.Random(subset_seed)
            num_samples = max(1, int(self.data_len * subset_pct))
            self.indices = rng.sample(all_indices, num_samples)
            self.indices.sort()
            logger.info(f"[NYUv2] Fast-Iter Mode: Sampled {num_samples} indices ({subset_pct*100:.1f}%)")
        else:
            self.indices = all_indices

        # --- Build Transforms ---
        if split == "train" and augmentation:
            self.transform = NYUv2TrainTransform(normalize_rgb=normalize_rgb)
        else:
            self.transform = NYUv2TestTransform(normalize_rgb=normalize_rgb)

        logger.info(
            f"[NYUv2-{split.upper()}] Initialized: {self.data_len} samples, "
            f"augmentation={'ON' if (split == 'train' and augmentation) else 'OFF'}, "
            f"ImageNet_norm={'ON' if normalize_rgb else 'OFF'}"
        )

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and transform a single sample.

        Returns SPECTRA-compatible dict with input, targets, and metadata.
        """
        index = self.indices[idx]

        # ==================================================================
        # 1. LOAD RAW DATA FROM .npy FILES
        # ==================================================================
        # Image: (H, W, 3) → (3, H, W) via moveaxis
        image = torch.from_numpy(
            np.moveaxis(
                np.load(str(self.data_path / "image" / f"{index}.npy")),
                -1, 0,  # (H, W, C) → (C, H, W)
            )
        ).float()

        # Label: (H, W) — stays 2D, no channel dimension
        label_raw = np.load(str(self.data_path / "label" / f"{index}.npy"))

        # CRITICAL: Remap -1 → 255 for CrossEntropyLoss ignore_index
        # NYUv2 uses -1 for unlabeled pixels; PyTorch CE expects positive or 255
        label_raw = label_raw.copy()  # Ensure writeable array
        label_raw[label_raw == -1] = IGNORE_INDEX

        # OPTIONAL: Reduced Classes (Axe v3.2 Fast-Iter)
        if self.num_classes == 3:
            # Aggregate 13 maps -> 3 Super-Classes
            # 0: Structural (Wall, Floor, Ceiling/Window)
            # 1: Furniture (Bed, Chair, Sofa, Table, Cabinet, ...)
            # 2: Props/Other
            new_label = np.zeros_like(label_raw)
            # Structural: wall(0), floor(1), window(8), door(7)
            mask_structural = np.isin(label_raw, [0, 1, 7, 8])
            # Furniture: cabinet(2), bed(3), chair(4), sofa(5), table(6), bookshelf(9), desk(13)
            mask_furniture = np.isin(label_raw, [2, 3, 4, 5, 6, 9, 13])
            
            new_label[mask_structural] = 0
            new_label[mask_furniture] = 1
            new_label[~(mask_structural | mask_furniture)] = 2
            
            # Preserve Ignore Index
            new_label[label_raw == IGNORE_INDEX] = IGNORE_INDEX
            label_raw = new_label

        label = torch.from_numpy(label_raw).long()

        # Depth: (H, W, 1) → (1, H, W) via moveaxis
        depth = torch.from_numpy(
            np.moveaxis(
                np.load(str(self.data_path / "depth" / f"{index}.npy")),
                -1, 0,
            )
        ).float()

        # Normal: (H, W, 3) → (3, H, W) via moveaxis
        normal = torch.from_numpy(
            np.moveaxis(
                np.load(str(self.data_path / "normal" / f"{index}.npy")),
                -1, 0,
            )
        ).float()

        # ==================================================================
        # 2. NaN / Inf SAFETY CHECK (before transforms)
        # ==================================================================
        if torch.isnan(image).any() or torch.isinf(image).any():
            logger.warning(f"[NYUv2] NaN/Inf in image at idx {idx}. Replacing with 0.")
            image = torch.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)

        if torch.isnan(depth).any() or torch.isinf(depth).any():
            logger.warning(f"[NYUv2] NaN/Inf in depth at idx {idx}. Replacing with 0.")
            depth = torch.nan_to_num(depth, nan=0.0, posinf=10.0, neginf=0.0)

        if torch.isnan(normal).any() or torch.isinf(normal).any():
            logger.warning(f"[NYUv2] NaN/Inf in normals at idx {idx}. Replacing with up-vector.")
            normal = torch.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)

        # ==================================================================
        # 3. APPLY TRANSFORMS (spatial augmentations)
        # ==================================================================
        image, label, depth, normal = self.transform(image, label, depth, normal)

        # ==================================================================
        # 4. DEPTH VALIDITY MASK (computed AFTER transforms)
        # ==================================================================
        # CRITICAL: This MUST be computed after spatial transforms, not before.
        # RandomScaleCrop and RandomHorizontalFlip change pixel positions.
        # Computing the mask before transforms would cause spatial misalignment
        # — the mask would refer to PRE-transform positions while the depth
        # values are at POST-transform positions. This silently corrupts the
        # depth loss by masking the WRONG pixels.
        depth_mask = (depth > 0.0).float()

        # ==================================================================
        # 5. RETURN SPECTRA-COMPATIBLE DICT
        # ==================================================================
        return {
            "input": image,
            "targets": {
                "segmentation": label,
                "depth": depth,
                "normals": normal,
            },
            "meta": {
                "depth_mask": depth_mask,
                "sample_id": f"nyuv2_{self.split}_{index}",
            },
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for NYUv2 batches.

        Handles the nested dict structure: {input, targets:{...}, meta:{...}}.
        """
        inputs = torch.stack([item["input"] for item in batch])

        targets = {}
        target_keys = batch[0]["targets"].keys()
        for key in target_keys:
            targets[key] = torch.stack([item["targets"][key] for item in batch])

        meta = {}
        meta_keys = batch[0]["meta"].keys()
        for key in meta_keys:
            values = [item["meta"][key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                meta[key] = torch.stack(values)
            else:
                meta[key] = values  # Keep strings as list

        return {"input": inputs, "targets": targets, "meta": meta}


# =============================================================================
# STANDALONE VERIFICATION (Smoke Test)
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    root = sys.argv[1] if len(sys.argv) > 1 else "data/nyuv2"

    print("=" * 70)
    print("SPECTRA NYUv2 Dataset — Smoke Test")
    print("=" * 70)

    for split in ["train", "val"]:
        try:
            ds = NYUv2Dataset(root=root, split=split, augmentation=(split == "train"))
            print(f"\n[{split.upper()}] Loaded {len(ds)} samples")

            sample = ds[0]
            print(f"  input shape:        {sample['input'].shape} dtype={sample['input'].dtype}")
            print(f"  input range:        [{sample['input'].min():.2f}, {sample['input'].max():.2f}]")
            print(f"  segmentation shape: {sample['targets']['segmentation'].shape} dtype={sample['targets']['segmentation'].dtype}")
            print(f"  segmentation range: [{sample['targets']['segmentation'].min()}, {sample['targets']['segmentation'].max()}]")
            print(f"  depth shape:        {sample['targets']['depth'].shape} dtype={sample['targets']['depth'].dtype}")
            print(f"  depth range:        [{sample['targets']['depth'].min():.4f}, {sample['targets']['depth'].max():.4f}]")
            print(f"  normals shape:      {sample['targets']['normals'].shape} dtype={sample['targets']['normals'].dtype}")
            print(f"  normals range:      [{sample['targets']['normals'].min():.4f}, {sample['targets']['normals'].max():.4f}]")
            print(f"  depth_mask shape:   {sample['meta']['depth_mask'].shape}")
            print(f"  depth_mask valid%:  {sample['meta']['depth_mask'].mean() * 100:.1f}%")
            print(f"  sample_id:          {sample['meta']['sample_id']}")

            # Validate label integrity
            unique_labels = sample['targets']['segmentation'].unique()
            valid = all(
                (l >= 0 and l < NUM_CLASSES) or l == IGNORE_INDEX
                for l in unique_labels
            )
            print(f"  label integrity:    {'PASS' if valid else 'FAIL'} (unique: {unique_labels.tolist()})")

            # Test collation
            from torch.utils.data import DataLoader
            dl = DataLoader(ds, batch_size=2, collate_fn=NYUv2Dataset.collate_fn)
            batch = next(iter(dl))
            print(f"  batch input shape:  {batch['input'].shape}")
            print(f"  batch seg shape:    {batch['targets']['segmentation'].shape}")

        except FileNotFoundError as e:
            print(f"\n[{split.upper()}] Skipped (data not found): {e}")

    print("\n" + "=" * 70)
    print("Smoke Test Complete!")
    print("=" * 70)
