"""
NYUv2 Multi-Task Dense Prediction Dataset.

Description:
    Loads the NYUv2 dataset from LMDB storage (materialized by nyuv2_lmdb_sota.py)
    and returns SPECTRA-compatible batch dictionaries with three task targets:
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
    - Fork safety: LMDB env re-initialized per worker process
    - Dtype enforcement: segmentation is ALWAYS int64 (CrossEntropyLoss safe)
    - NaN/Inf trapping: depth and normals checked for corruption
    - Ignore class remapping: invalid labels → 255 for cross-entropy ignore_index
    - Label clamping: ensures all label values are in {0..12, 255}

References:
    - Silberman et al. "Indoor Segmentation and Support Inference from RGBD Images" (ECCV 2012)
    - Liu et al. "End-to-End Multi-Task Learning with Attention" (CVPR 2019) [MTAN]
    - Standard 13-class split: https://github.com/lorenmt/mtan
"""

from __future__ import annotations

import os
import lmdb
import json
import logging
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


def resolve_nyuv2_root(root: str | Path) -> Path:
    """
    Resolve the NYUv2 LMDB root across supported on-disk layouts.

    Supported:
      1. root/train/data.lmdb + root/train_index.json
      2. root/data/train/data.lmdb + root/data/train_index.json
    """
    root_path = Path(root)
    candidates = [root_path, root_path / "data"]

    for candidate in candidates:
        if (candidate / "train").exists() or (candidate / "val").exists():
            return candidate

    return root_path


# =============================================================================
# CORE DATASET
# =============================================================================

class NYUv2Dataset(Dataset):
    """
    NYUv2 Multi-Task Learning Dataset.

    Loads pre-processed data from LMDB storage (materialized by nyuv2_lmdb_sota.py):
        root/
            train/data.lmdb
            val/data.lmdb
            train_index.json
            val_index.json

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
        subset_pct: Fraction of data to use (1.0 = all).
        subset_seed: Seed for subset sampling reproducibility.
        num_classes: Number of semantic classes (default 13).
    """

    def __init__(
        self,
        root: str = "datasets/nyuv2_lmdb",
        split: str = "train",
        augmentation: bool = True,
        normalize_rgb: bool = False,
        subset_pct: float = 1.0,
        subset_seed: int = 42,
        subset_file: Optional[str] = None,
        subset_id: Optional[str] = None,
        num_classes: int = 13,
    ):
        super().__init__()

        self.root = resolve_nyuv2_root(root)
        self.split = "val" if split in ["validation", "val", "test"] else "train"
        self.num_classes = num_classes
        
        # Multi-directory resolution:
        # The structure places data.lmdb inside split-specific subdirs
        self.split_dir = self.root / self.split
        self.lmdb_path = self.split_dir / "data.lmdb"
        self.index_path = self.root / f"{self.split}_index.json"
        
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"[NYUv2] LMDB missing at: {self.lmdb_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"[NYUv2] Index missing at: {self.index_path}")
            
        with open(self.index_path, "r") as f:
            self.manifest = json.load(f)
            
        # The new schema uses "episodes" as the primary list
        self.samples = self.manifest.get("episodes", [])
        self.data_len = len(self.samples)
        
        # Global stats are stored inside "metadata"
        self.metadata = self.manifest.get("metadata", {})
        self.stats = self.metadata.get("stats", {})
        self.storage = self.metadata.get("storage", {})
        self.sanitized = self.metadata.get("sanitized", {})
        self.image_layout = self.storage.get("image_layout", "hwc")
        self.depth_layout = self.storage.get("depth_layout", "hwc")
        self.normal_layout = self.storage.get("normal_layout", "hwc")
        self.label_dtype = self.storage.get("label_dtype", "uint8")
        self.skip_runtime_nan_sanitize = bool(self.sanitized.get("runtime_safe_finite", False))
        self.subset_file = subset_file
        self.subset_id = subset_id
        self.subset_metadata: Dict[str, Any] = {}
        
        if self.data_len == 0:
            logger.warning(f"[NYUv2] Split {self.split} index is EMPTY.")

        # Fork-safety
        self._lmdb_env = None
        self._parent_pid = os.getpid()

        # --- Build Index (Subset support) ---
        all_indices = list(range(self.data_len))
        
        if self.split == "train" and subset_file:
            subset_path = Path(subset_file)
            if not subset_path.is_absolute():
                subset_path = subset_path.resolve()
            if not subset_path.exists():
                raise FileNotFoundError(f"[NYUv2] Subset file missing at: {subset_path}")

            with subset_path.open("r", encoding="utf-8") as handle:
                subset_payload = json.load(handle)

            indices = subset_payload.get("indices")
            if not isinstance(indices, list) or not indices:
                raise ValueError(f"[NYUv2] Subset file '{subset_path}' does not contain a non-empty 'indices' list.")

            invalid = [idx for idx in indices if not isinstance(idx, int) or idx < 0 or idx >= self.data_len]
            if invalid:
                raise ValueError(
                    f"[NYUv2] Subset file '{subset_path}' contains invalid train indices; "
                    f"first invalid entries: {invalid[:5]}"
                )

            self.indices = sorted(indices)
            self.subset_metadata = subset_payload.get("metadata", {})
            if not self.subset_id:
                self.subset_id = subset_payload.get("subset_id")
            logger.info(
                "[NYUv2] Paper subset active: %s (%d samples)",
                self.subset_id or subset_path.name,
                len(self.indices),
            )
        elif 0.0 < subset_pct < 1.0:
            import random
            rng = random.Random(subset_seed)
            num_samples = max(1, int(self.data_len * subset_pct))
            self.indices = rng.sample(all_indices, num_samples)
            self.indices.sort()
            logger.info(f"[NYUv2] Fast-Iter Mode: Sampled {num_samples} indices ({subset_pct*100:.1f}%)")
        else:
            self.indices = all_indices

        # --- Build Transforms ---
        if self.split == "train" and augmentation:
            self.transform = NYUv2TrainTransform(normalize_rgb=normalize_rgb)
        else:
            self.transform = NYUv2TestTransform(normalize_rgb=normalize_rgb)

        logger.info(
            f"[NYUv2-{self.split.upper()}] Initialized: {self.data_len} samples, "
            f"augmentation={'ON' if (self.split == 'train' and augmentation) else 'OFF'}, "
            f"ImageNet_norm={'ON' if normalize_rgb else 'OFF'}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _init_lmdb(self):
        curr_pid = os.getpid()
        if self._lmdb_env is not None and curr_pid != self._parent_pid:
            self._lmdb_env = None
            self._parent_pid = curr_pid

        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False,
                max_spare_txns=1,
            )

    def _read_sample_bytes(self, sample_meta: Dict[str, Any]) -> Tuple[bytes, bytes, bytes, bytes]:
        """
        Fetch all modalities for one sample under a single read transaction.

        The previous path opened four independent read transactions per sample.
        LMDB read transactions are cheap but not free; collapsing them to one
        materially reduces Python and LMDB overhead in the dataloader hot path.
        """
        self._init_lmdb()
        with self._lmdb_env.begin(write=False) as txn:
            image = txn.get(sample_meta["image_key"].encode("ascii"))
            label = txn.get(sample_meta["label_key"].encode("ascii"))
            depth = txn.get(sample_meta["depth_key"].encode("ascii"))
            normal = txn.get(sample_meta["normal_key"].encode("ascii"))

        if image is None:
            raise KeyError(f"LMDB Key failure: {sample_meta['image_key']}")
        if label is None:
            raise KeyError(f"LMDB Key failure: {sample_meta['label_key']}")
        if depth is None:
            raise KeyError(f"LMDB Key failure: {sample_meta['depth_key']}")
        if normal is None:
            raise KeyError(f"LMDB Key failure: {sample_meta['normal_key']}")

        return image, label, depth, normal

    @staticmethod
    def _decode_uint8_image(buffer: bytes, hw: List[int], layout: str = "hwc") -> torch.Tensor:
        height, width = hw
        base = torch.frombuffer(buffer, dtype=torch.uint8)
        if layout == "chw":
            image = base.view(3, height, width)
        else:
            image = base.view(height, width, 3).permute(2, 0, 1).contiguous()
        return image.to(dtype=torch.float32).div_(255.0)

    @staticmethod
    def _decode_uint8_label(buffer: bytes, hw: List[int]) -> torch.Tensor:
        height, width = hw
        return torch.frombuffer(buffer, dtype=torch.uint8).view(height, width)

    @staticmethod
    def _decode_float16_map(buffer: bytes, hw: List[int], channels: int, layout: str = "hwc") -> torch.Tensor:
        height, width = hw
        base = torch.frombuffer(buffer, dtype=torch.float16)
        if layout == "chw":
            tensor = base.view(channels, height, width)
        else:
            tensor = base.view(height, width, channels).permute(2, 0, 1).contiguous()
        return tensor.to(dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.indices[idx]
        sample_meta = self.samples[global_idx]
        hw = sample_meta["shape_hw"]

        # 1. Fetch & Deserialize
        img_bytes, lbl_bytes, depth_bytes, norm_bytes = self._read_sample_bytes(sample_meta)

        image = self._decode_uint8_image(img_bytes, hw, layout=self.image_layout)
        label = self._decode_uint8_label(lbl_bytes, hw)
        depth = self._decode_float16_map(depth_bytes, hw, channels=1, layout=self.depth_layout)
        normal = self._decode_float16_map(norm_bytes, hw, channels=3, layout=self.normal_layout)

        # 2. Safety Checks
        if not self.skip_runtime_nan_sanitize:
            depth = torch.nan_to_num(depth)
            normal = torch.nan_to_num(normal)

        # Label integrity: convert to int64 BEFORE masked_fill_ to avoid uint8 truncation
        label = label.to(dtype=torch.long)
        label.masked_fill_(label >= self.num_classes, IGNORE_INDEX)

        # 3. Apply Transforms
        image, label, depth, normal = self.transform(image, label, depth, normal)

        # 4. Meta / Mask
        depth_mask = (depth > 0.0).float()

        return {
            "input": image,
            "targets": {
                "segmentation": label,
                "depth": depth,
                "normals": normal,
            },
            "meta": {
                "depth_mask": depth_mask,
                "sample_id": f"nyuv2_{self.split}_{global_idx}",
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

    root = sys.argv[1] if len(sys.argv) > 1 else "datasets/nyuv2"

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
