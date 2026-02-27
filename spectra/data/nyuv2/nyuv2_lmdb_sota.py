"""
spectra/data/nyuv2/nyuv2_lmdb_sota.py
--------------------------------------------------------------------------------
SPECTRA SOTA Data Ingestion Pipeline (v5.0 — "Axe Specification")
Author: SPECTRA Research Team
Status: Production / Multi-Stage Hardened

Description:
    Processes the NYUv2 dataset (tanganke/nyuv2) into a high-fidelity LMDB
    storage format. This pipeline follows the "Axe" philosophy: multi-stage
    staging, atomic materialization, and automated cleanup.

STAGES:
    1. INGESTION: Download (via HF cache) and write to LMDB in a unified pipeline.
    2. CLEANUP: Purge the staging directory and reclaim space.

Safety Guarantees:
    - DiskGuard: Pre-checks available space (minimum 6GB).
    - FP64 Welford: Numerically stable global Mean/Std calculation.
    - Atomic Commits: LMDB transactions committed every 100 samples.
    - Architectural Parity: Matches the high-fidelity clinical branch.
"""

import os
import sys
import json
import lmdb
import shutil
import logging
import hashlib
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple

# ==============================================================================
# AXE v6.4: ABSOLUTE PROJECT ISOLATION (HuggingFace Cache Guard)
# ==============================================================================
# Force all HF activity into the project-local staging directory.
# This MUST be set before engine initialization.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
STAGING_DIR = _PROJECT_ROOT / "data" / "staging_nyuv2"
os.environ["HF_HOME"] = str(STAGING_DIR.absolute())
os.environ["HF_DATASETS_CACHE"] = str(STAGING_DIR.absolute())
os.environ["HUGGINGFACE_HUB_CACHE"] = str(STAGING_DIR.absolute())
os.environ["HF_HUB_CACHE"] = str(STAGING_DIR.absolute())

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_REPO = "tanganke/nyuv2"
OUTPUT_DIR = Path("data/nyuv2_lmdb")
# STAGING_DIR is now globally managed above
LMDB_MAP_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
COMMIT_FREQ = 100
MIN_FREE_SPACE_GB = 3.0
CLEANUP_STAGING = True  # Toggle to False to keep staged data for future runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SPECTRA_AXE_v5")

# ==============================================================================
# 1. UTILITY ENGINES
# ==============================================================================

class WelfordSpatialEngine:
    """Online statistics for spatial feature maps (Mean/Std in FP64)."""
    def __init__(self, channels: int):
        self.n = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)
    
    def update(self, x_hwc: np.ndarray):
        """Update with an [H, W, C] array using vectorized Welford's."""
        flat = x_hwc.reshape(-1, x_hwc.shape[-1]).astype(np.float64)
        m = flat.shape[0]
        if m == 0: return

        # Batch statistics
        batch_mean = np.mean(flat, axis=0)
        batch_m2 = np.sum((flat - batch_mean)**2, axis=0)

        n_new = self.n + m
        delta = batch_mean - self.mean
        
        # Parallel Update Rule
        self.mean = self.mean + delta * (m / n_new)
        self.m2 = self.m2 + batch_m2 + (delta**2) * (self.n * m / n_new)
        self.n = n_new
            
    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            return self.mean.astype(np.float32), np.ones_like(self.mean, dtype=np.float32)
        variance = self.m2 / (self.n - 1)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0 
        return self.mean.astype(np.float32), std.astype(np.float32)

def ensure_contiguous_hwc(array: np.ndarray) -> np.ndarray:
    """Aligns array to [H, W, C] format. Transposes if [C, H, W] is detected."""
    if array.ndim == 3:
        # Detected [C, H, W] format (C <= 4)
        if array.shape[0] <= 4 and array.shape[0] < array.shape[1]:
            array = np.transpose(array, (1, 2, 0))
    return np.ascontiguousarray(array)

class PhysicsEngine:
    """
    SOTA Geometric & Numerical Validation Engine.
    Handles depth clamping, normal normalization, and shape consistency.
    """
    DEPTH_MAX = 10.0
    DEPTH_MIN = 0.0
    EPS = 1e-8

    @staticmethod
    def validate_shape_consistency(img: np.ndarray, depth: np.ndarray, normal: np.ndarray, label: np.ndarray):
        """Ensures all modalities share the same spatial dimensions."""
        h, w = img.shape[:2]
        if depth.shape[:2] != (h, w) or normal.shape[:2] != (h, w) or label.shape[:2] != (h, w):
            raise ValueError(f"Shape Mismatch: img {(h, w)}, depth {depth.shape[:2]}, normal {normal.shape[:2]}, label {label.shape[:2]}")

    @staticmethod
    def process_depth(depth: np.ndarray) -> np.ndarray:
        """Clamps depth to [0, 10] meters, handles NaNs, and ensures contiguous HWC."""
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=PhysicsEngine.DEPTH_MAX, neginf=PhysicsEngine.DEPTH_MIN)
        depth = ensure_contiguous_hwc(depth)
        if depth.ndim == 2: depth = depth[:, :, np.newaxis]
        # In NYUv2, 0 often means invalid/missing. We keep it as 0 but clamp the upper bound.
        return np.clip(depth, PhysicsEngine.DEPTH_MIN, PhysicsEngine.DEPTH_MAX)

    @staticmethod
    def process_normal(normal: np.ndarray) -> np.ndarray:
        """Enforces L2 unit-length normalization on surface normals and handles NaNs."""
        normal = np.nan_to_num(normal.astype(np.float32), nan=0.0)
        normal = ensure_contiguous_hwc(normal)
        mag = np.linalg.norm(normal, axis=-1, keepdims=True)
        # Handle zero vectors to avoid division by zero. If mag < EPS, we return zero vector.
        # SOTA: Could return [0, 0, 1] as default, but 0 is safer for gradient masks.
        safe_normal = np.where(mag > PhysicsEngine.EPS, normal / (mag + PhysicsEngine.EPS), 0.0)
        return safe_normal.astype(np.float32)

class StatsReservoir:
    """Reservoir sampling for robust quantile estimation (P01/P99)."""
    def __init__(self, channels: int, max_size: int = 1000):
        self.max_size = max_size
        self.reservoir = []
        self.channels = channels

    def update(self, x_hwc: np.ndarray):
        """Adds a random spatial subset to the reservoir with a memory compaction guard."""
        # Take a random 1% spatial sample
        flat = x_hwc.reshape(-1, self.channels)
        n_points = max(1, len(flat) // 100)
        indices = np.random.choice(len(flat), n_points, replace=False)
        self.reservoir.append(flat[indices])

        # Memory compaction: if reservoir grows too large, downsample it
        if len(self.reservoir) > self.max_size:
            big_block = np.concatenate(self.reservoir, axis=0)
            # Resample back to a manageable size (e.g., 500k points)
            keep_idx = np.random.choice(len(big_block), 500_000, replace=False)
            self.reservoir = [big_block[keep_idx]]

    def get_quantiles(self) -> Tuple[List[float], List[float]]:
        """Returns P01 and P99 per channel with NaN resilience."""
        if not self.reservoir:
            return [0.0] * self.channels, [1.0] * self.channels
        
        data = np.concatenate(self.reservoir, axis=0)
        # Filtering out NaNs just in case they slipped through earlier stages
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) == 0:
            return [0.0] * self.channels, [1.0] * self.channels

        p01 = np.percentile(data, 1, axis=0).tolist()
        p99 = np.percentile(data, 99, axis=0).tolist()
        return p01, p99

class DiskGuard:
    """Safety check for disk availability."""
    @staticmethod
    def check_space(min_gb: float = MIN_FREE_SPACE_GB):
        _, _, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        if free_gb < min_gb:
            logger.critical(f"DISK SPACE FAILURE: {free_gb:.1f}GB available, need {min_gb}GB.")
            sys.exit(1)
        logger.info(f"Disk Guard: {free_gb:.1f}GB available. Space check PASSED.")

# ==============================================================================
# 2. INGESTION ENGINE
# ==============================================================================

class QualityIngestionEngine:
    def __init__(self):
        DiskGuard.check_space()
        
        # Axe v6.1: Multi-Split Foundation
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.envs = {}
        self.indices = {} # Dynamic split initialization
        
        # Statistics & Reservoirs are split-aware but usually aggregated from Train
        self.stats = {
            "image": WelfordSpatialEngine(3),
            "depth": WelfordSpatialEngine(1),
            "normal": WelfordSpatialEngine(3)
        }
        self.reservoirs = {
            "image": StatsReservoir(3),
            "depth": StatsReservoir(1),
            "normal": StatsReservoir(3)
        }
        self.total_written = 0

    def _init_split_env(self, target_name: str):
        """Initializes a specific LMDB environment for a split."""
        split_dir = OUTPUT_DIR / target_name
        split_dir.mkdir(parents=True, exist_ok=True)
        lmdb_path = split_dir / "data.lmdb"
        
        # Optimize LMDB for production: MAPASYNC for speed. 
        # Note: writemap=True is disabled for Windows filesystem stability.
        env = lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE, subdir=False, 
                        map_async=True, writemap=False)
        self.envs[target_name] = env
        return env


    def process_split(self, ds_split: Any, split_name: str, limit: Optional[int] = None):
        """Stage 2: Process and write to LMDB specific to the split (Streaming Optimized)."""
        # Intelligently map HF split names to SPECTRA standard names
        sn_lower = split_name.lower()
        if "train" in sn_lower:
            target_name = "train"
            total_estimate = 795
        elif any(x in sn_lower for x in ["val", "test", "eval"]):
            target_name = "val"
            total_estimate = 654
        else:
            target_name = split_name # Fallback
            total_estimate = None
            
        logger.info(f"Stage 2: Streaming Split: {split_name} -> {target_name} (Limit: {limit})")
        
        if target_name not in self.indices:
            self.indices[target_name] = []

        # Initialize sub-environment if not already active
        env = self.envs.get(target_name) or self._init_split_env(target_name)
        txn = env.begin(write=True)
        
        pbar = tqdm(total=limit if limit else total_estimate, desc=f"Streaming {target_name}")
        try:
            for i, sample in enumerate(ds_split):
                if limit is not None and i >= limit:
                    break
                # --- 1. RAW EXTRACTION & STANDARDIZATION ---
                img = np.array(sample["image"])
                lbl = np.array(sample["segmentation"]).astype(np.uint8)
                depth_raw = np.array(sample["depth"])
                norm_raw = np.array(sample["normal"])

                # --- 2. PHYSICS VALIDATION ---
                # Standardize to [H, W, C] before validation
                img = ensure_contiguous_hwc(img)
                depth_raw = ensure_contiguous_hwc(depth_raw)
                norm_raw = ensure_contiguous_hwc(norm_raw)
                lbl = np.ascontiguousarray(lbl) # Segmentation is usually [H, W]

                # Enforce spatial consistency
                PhysicsEngine.validate_shape_consistency(img, depth_raw, norm_raw, lbl)
                
                # Process modalities
                if img.dtype != np.uint8:
                    if img.max() <= 1.01: img = (img * 255)
                    img = img.astype(np.uint8)
                
                depth = PhysicsEngine.process_depth(depth_raw)
                norm = PhysicsEngine.process_normal(norm_raw)
                
                # Cast to high-fidelity storage formats
                depth_fp16 = depth.astype(np.float16)
                norm_fp16 = norm.astype(np.float16)

                # --- 3. STATISTICS & RESERVOIR (Train Only) ---
                if target_name == "train":
                    img_normalized = img.astype(np.float32) / 255.0
                    self.stats["image"].update(img_normalized)
                    self.stats["depth"].update(depth)
                    self.stats["normal"].update(norm)
                    
                    # Update reservoirs for quantile estimation
                    self.reservoirs["image"].update(img_normalized)
                    self.reservoirs["depth"].update(depth)
                    self.reservoirs["normal"].update(norm)
                
                # --- 4. ATOMIC SERIALIZATION ---
                keys = {
                    "img": f"{target_name}_{i}_img",
                    "lbl": f"{target_name}_{i}_lbl",
                    "dpt": f"{target_name}_{i}_dpt",
                    "nrm": f"{target_name}_{i}_nrm"
                }
                
                txn.put(keys["img"].encode('ascii'), img.tobytes())
                txn.put(keys["lbl"].encode('ascii'), lbl.tobytes())
                txn.put(keys["dpt"].encode('ascii'), depth_fp16.tobytes())
                txn.put(keys["nrm"].encode('ascii'), norm_fp16.tobytes())
                
                self.indices[target_name].append({
                    "idx": i,
                    "image_key": keys["img"],
                    "label_key": keys["lbl"],
                    "depth_key": keys["dpt"],
                    "normal_key": keys["nrm"],
                    "shape_hw": [img.shape[0], img.shape[1]]
                })
                
                self.total_written += 1
                if (i + 1) % COMMIT_FREQ == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                pbar.update(1)
                
            txn.commit()
            pbar.close()
        except Exception as e:
            logger.error(f"FATAL ERROR during split {target_name} ingestion: {e}")
            txn.abort()
            raise e
        
        logger.info(f"Stage 2: {target_name} Ingestion COMPLETE (Total: {len(self.indices[target_name])})")

    def finalize(self):
        """Generate separate Manifests and finalize all LMDB environments."""
        logger.info("Stage 2: Finalizing statistics and separate manifests...")
        img_m, img_s = self.stats["image"].finalize()
        depth_m, depth_s = self.stats["depth"].finalize()
        norm_m, norm_s = self.stats["normal"].finalize()
        
        # Compute quantiles from reservoirs
        img_p01, img_p99 = self.reservoirs["image"].get_quantiles()
        depth_p01, depth_p99 = self.reservoirs["depth"].get_quantiles()
        norm_p01, norm_p99 = self.reservoirs["normal"].get_quantiles()

        metadata = {
            "version": "SPECTRA-NYUv2-LMDB-v6.1-AXE",
            "stats": {
                "image": {
                    "mean": img_m.tolist(), "std": img_s.tolist(),
                    "p01": img_p01, "p99": img_p99
                },
                "depth": {
                    "mean": depth_m.tolist(), "std": depth_s.tolist(),
                    "p01": depth_p01, "p99": depth_p99
                },
                "normal": {
                    "mean": norm_m.tolist(), "std": norm_s.tolist(),
                    "p01": norm_p01, "p99": norm_p99
                },
            }
        }

        # Save separate index files for each split
        for split_name, entries in self.indices.items():
            manifest = {
                "metadata": metadata,
                "episodes": entries
            }
            out_path = OUTPUT_DIR / f"{split_name}_index.json"
            with open(out_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Manifest saved: {out_path}")

        # Close all environments
        for name, env in self.envs.items():
            env.close()
            logger.info(f"LMDB Environment closed: {name}")
            
        logger.info(f"Stage 2: SUCCESS. Multi-split LMDB built at {OUTPUT_DIR.absolute()}")

def cleanup():
    """Stage 3: Cleanup."""
    logger.info("Stage 3: Automated Cleanup...")
    if CLEANUP_STAGING and STAGING_DIR.exists():
        logger.info(f"Removing Staging Directory: {STAGING_DIR}")
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
    elif not CLEANUP_STAGING:
        logger.info(f"CLEANUP_STAGING is OFF. Staging kept at {STAGING_DIR}")
    
    logger.info("Hint: Run 'huggingface-cli delete-cache' to reclaim additional global space.")
    logger.info("Stage 3: COMPLETED.")

# ==============================================================================
# 3. VALIDATION ENGINE
# ==============================================================================

def validate_lmdb():
    """Reads back samples from all LMDB splits to verify integrity."""
    logger.info("Stage 2.5: Verifying LMDB Integrity across splits...")
    
    for split in ["train", "val"]:
        split_dir = OUTPUT_DIR / split
        if not (split_dir / "data.lmdb").exists():
            logger.warning(f"Validation: {split} LMDB not found at {split_dir}")
            continue

        env = lmdb.open(str(split_dir / "data.lmdb"), readonly=True, subdir=False)
        with env.begin() as txn:
            for suffix in ["img", "lbl", "dpt", "nrm"]:
                key = f"{split}_0_{suffix}"
                data = txn.get(key.encode())
                if data:
                    logger.info(f"Integrity Check [{split}]: {key} found ({len(data)} bytes).")
                else:
                    logger.warning(f"Integrity Check [{split}]: {key} NOT found.")
        env.close()
    
    logger.info("Stage 2.5: Validation COMPLETED.")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SPECTRA NYUv2 Ingestion")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per split for testing")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("SPECTRA MULTI-STAGE INGESTION (v6.0 — Axe Specification)")
    logger.info("="*60)
    
    # Fixed seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logger.info(f"Global seed set to {SEED}")

    engine = QualityIngestionEngine()
    success = False
    
    try:
        # Stage 1: Dataset Acquisition (Streaming mode for speed & low overhead)
        logger.info(f"Phase 1: Streaming dataset from {DATASET_REPO}...")
        ds = load_dataset(DATASET_REPO, streaming=True)
        
        # Stage 2: Materialization
        available_splits = list(ds.keys())
        logger.info(f"Phase 2: Ingesting streaming splits: {available_splits}")
        
        for split_key in available_splits:
            engine.process_split(ds[split_key], split_key, limit=args.limit)
        
        success = True
    except (Exception, KeyboardInterrupt) as e:
        logger.critical(f"UNRECOVERABLE FAILURE OR ABORT: {type(e).__name__}: {e}")
        # Explicitly close envs without finalizing manifest if failed
        for name, env in engine.envs.items():
            env.close()
        raise
    finally:
        if success:
            engine.finalize()
        else:
            logger.warning("Execution did not complete. Manifests were NOT generated.")

    # Stage 2.5: Validation
    validate_lmdb()
    
    # Stage 3
    cleanup()
    
    logger.info("="*60)
    logger.info("MISSION COMPLETE. NYUv2 is ready for high-fidelity training.")
    logger.info("="*60)

if __name__ == "__main__":
    main()
