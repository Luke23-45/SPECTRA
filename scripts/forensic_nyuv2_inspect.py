"""
scripts/forensic_nyuv2_inspect.py
-----------------------------------
Forensic Auditor for NYUv2 LMDB materialization.

Performs deep mathematical and visual checks on the materialized data
to ensure zero-corruption and alignment with Phase 1 specs.
"""

import os
import json
import lmdb
import numpy as np
import torch
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Forensic-Auditor")

def forensic_audit(split="train", sample_idx=0):
    root = Path("datasets/nyuv2_lmdb")
    split_dir = root / split
    lmdb_path = split_dir / "data.lmdb"
    index_path = root / f"{split}_index.json"

    if not lmdb_path.exists():
        logger.error(f"LMDB missing: {lmdb_path}")
        return

    # 1. Load Index
    with open(index_path, "r") as f:
        manifest = json.load(f)
    samples = manifest.get("episodes", [])
    if sample_idx >= len(samples):
        logger.error(f"Sample index {sample_idx} out of range (max {len(samples)-1})")
        return
    
    sample_meta = samples[sample_idx]
    logger.info(f"Auditing Sample {sample_idx} [{split}]")

    # 2. Extract from LMDB
    env = lmdb.open(str(lmdb_path), readonly=True, subdir=False, lock=False)
    with env.begin() as txn:
        img_bin = txn.get(sample_meta["image_key"].encode())
        lbl_bin = txn.get(sample_meta["label_key"].encode())
        dpt_bin = txn.get(sample_meta["depth_key"].encode())
        nrm_bin = txn.get(sample_meta["normal_key"].encode())

    env.close()

    # 3. Deserialize & Forensic Check
    h, w = sample_meta["shape_hw"]
    
    # Image [H, W, 3] uint8
    img = np.frombuffer(img_bin, dtype=np.uint8).reshape(h, w, 3)
    # Label [H, W] uint8
    lbl = np.frombuffer(lbl_bin, dtype=np.uint8).reshape(h, w)
    # Depth [H, W, 1] fp16 -> fp32
    dpt = np.frombuffer(dpt_bin, dtype=np.float16).reshape(h, w, 1).astype(np.float32)
    # Normals [H, W, 3] fp16 -> fp32
    nrm = np.frombuffer(nrm_bin, dtype=np.float16).reshape(h, w, 3).astype(np.float32)

    logger.info("--- Mathematical Integrity Check ---")
    
    # NaN/Inf Check
    for name, data in [("Image", img), ("Label", lbl), ("Depth", dpt), ("Normals", nrm)]:
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        status = "FAIL" if has_nan or has_inf else "PASS"
        logger.info(f"[{name}] NaN/Inf Check: {status}")
        if has_nan or has_inf: raise ValueError(f"Corruption detected in {name}!")

    # Value Range Checks
    logger.info(f"[Depth] Range: [{dpt.min():.2f}, {dpt.max():.2f}] (Expected: [0, 10])")
    
    # Normal Vector Integrity (Unit Length)
    # We ignore the zero-vectors which indicate invalid regions
    mags = np.linalg.norm(nrm, axis=-1)
    valid_mask = mags > 1e-4
    if valid_mask.any():
        avg_mag = mags[valid_mask].mean()
        logger.info(f"[Normals] Mean magnitude of valid pixels: {avg_mag:.4f} (Expected: ~1.0)")
        if abs(avg_mag - 1.0) > 0.05:
            logger.warning("[Normals] POTENTIAL UNIT LENGTH VIOLATION!")

    # 4. Visual Proof-of-Life
    # Normalize for visualization
    img_vis = img
    lbl_vis = (lbl.astype(np.float32) * (255.0 / 13.0)).astype(np.uint8)
    lbl_vis = cv2.applyColorMap(lbl_vis, cv2.COLORMAP_JET)
    
    dpt_norm = (dpt - dpt.min()) / (dpt.max() - dpt.min() + 1e-8)
    dpt_vis = (dpt_norm * 255).astype(np.uint8)
    dpt_vis = cv2.applyColorMap(dpt_vis, cv2.COLORMAP_MAGMA)
    
    # Normals: [-1, 1] -> [0, 255]
    nrm_vis = ((nrm + 1.0) / 2.0 * 255).astype(np.uint8)

    # Horizontal Montage
    montage = np.hstack([img_vis, lbl_vis, dpt_vis, nrm_vis])
    
    output_path = f"nyuv2_forensic_{split}_{sample_idx}.png"
    cv2.imwrite(output_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    logger.info(f"Audit visual saved to: {output_path}")

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    forensic_audit(split="train", sample_idx=idx)
    forensic_audit(split="val", sample_idx=idx)
