"""
scripts/materialize_nyuv2.py (v2.5)
----------------------------------
Axe v3.2 Optimized High-Fidelity Materialization.
- Force `float32` precision (saves 50% space vs float64 source).
- Maintains "not int" fidelity for images/depth/normals.
- Streaming iteration to prevent HF cache leaks.
"""

import os
import logging
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# CONFIG
DATASET_REPO = "tanganke/nyuv2"
OUTPUT_ROOT = Path("data/nyuv2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("SPECTRA-Data")

def materialize():
    logger.info("--- SPECTRA DATA PIPELINE v2.5 (Float32/Optimized) ---")
    
    # 1. Load Dataset in Streaming Mode
    ds = load_dataset(DATASET_REPO, streaming=True)
    
    for split_key in ["train", "val"]:
        logger.info(f"Processing split: {split_key}")
        
        split_dir = OUTPUT_ROOT / split_key
        for mod in ["image", "label", "depth", "normal"]:
            (split_dir / mod).mkdir(parents=True, exist_ok=True)
            
        count = 0
        for i, sample in enumerate(tqdm(ds[split_key], desc=f"Materializing {split_key}")):
            
            # --- IMAGE (Force float32) ---
            img = np.array(sample["image"]).astype(np.float32)
            if img.ndim == 3 and img.shape[0] <= 3:
                img = np.moveaxis(img, 0, -1)
            np.save(split_dir / "image" / f"{i}.npy", img)
            
            # --- LABEL (uint8 is fine for classification indices) ---
            lbl = np.array(sample["segmentation"]).astype(np.uint8)
            np.save(split_dir / "label" / f"{i}.npy", lbl)
            
            # --- DEPTH (Force float32) ---
            depth = np.array(sample["depth"]).astype(np.float32)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            if depth.ndim == 2:
                depth = depth[:, :, np.newaxis]
            np.save(split_dir / "depth" / f"{i}.npy", depth)
            
            # --- NORMAL (Force float32) ---
            norm = np.array(sample["normal"]).astype(np.float32)
            if norm.ndim == 3 and norm.shape[0] <= 3:
                norm = np.moveaxis(norm, 0, -1)
            np.save(split_dir / "normal" / f"{i}.npy", norm)
            
            count += 1

    logger.info(f"✔ Success. Data rooted at {OUTPUT_ROOT.absolute()}")
    logger.info("Fidelity: [Image/Depth/Normal: float32], [Label: uint8].")

if __name__ == "__main__":
    materialize()