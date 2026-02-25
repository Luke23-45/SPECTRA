"""
spectra/data/nyuv2/download.py
-------------------------------
Automated NYUv2 Data Acquisition (MTAN Standard Format).

Downloads pre-processed NYUv2 .npy files from the official MTAN Dropbox link.
This is the standard format used by all major MTL benchmarks (MTAN, LibMTL,
CAGrad, Nash-MTL, etc.)

Expected directory structure after download:
    {root}/
        train/
            image/   → 795 files (0.npy .. 794.npy)
            label/   → 795 files
            depth/   → 795 files
            normal/  → 795 files
        val/
            image/   → 654 files
            label/   → 654 files
            depth/   → 654 files
            normal/  → 654 files

Data Format (per .npy file):
    image:  (H, W, 3)  float32 — RGB pixels [0, 255]
    label:  (H, W)     int     — Semantic class indices {-1, 0..12}
    depth:  (H, W, 1)  float32 — Metric depth (meters)
    normal: (H, W, 3)  float32 — Unit surface normals (x, y, z)
"""

from __future__ import annotations

import os
import logging
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("spectra.data.nyuv2")

# Official MTAN preprocessed NYUv2 (Dropbox, ~420MB)
DROPBOX_URL = "https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=1"

# Expected file counts (hard-coded from the standard MTL benchmark split)
EXPECTED_COUNTS = {
    "train": 795,
    "val": 654,
}


def download_nyuv2(
    root: str = "data/nyuv2",
    url: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Download and extract NYUv2 dataset in MTAN format.

    Implements a tiered acquisition strategy (mirroring the clinical pipeline):
      Tier 0: Check if local data is valid and complete.
      Tier 1: Download from Dropbox if missing.

    Args:
        root: Target directory for the extracted dataset.
        url: Override URL for the dataset archive. Defaults to MTAN Dropbox.
        force: Re-download even if local data exists.

    Returns:
        Path to the root directory containing train/ and val/.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    root_path = Path(root)
    url = url or DROPBOX_URL

    # --- Tier 0: Local Integrity Check ---
    if not force and _verify_integrity(root_path):
        logger.info(f"[NYUv2] Local data verified at {root_path}")
        return root_path

    # --- Tier 1: Download ---
    logger.info(f"[NYUv2] Downloading from {url}...")
    root_path.mkdir(parents=True, exist_ok=True)
    zip_path = root_path / "nyuv2.zip"

    try:
        # Use urllib (stdlib) to avoid extra dependencies
        import urllib.request
        import shutil

        # Progress reporting for large downloads
        def _report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded * 100.0 / total_size)
                if block_num % 100 == 0:
                    logger.info(f"  Download progress: {pct:.1f}% ({downloaded // (1024*1024)}MB)")

        urllib.request.urlretrieve(url, str(zip_path), reporthook=_report_hook)
        logger.info(f"[NYUv2] Download complete: {zip_path}")

    except Exception as e:
        logger.critical(f"[NYUv2] Download FAILED: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Failed to download NYUv2 dataset: {e}") from e

    # --- Extract ---
    try:
        logger.info("[NYUv2] Extracting archive...")
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(root_path))
        logger.info("[NYUv2] Extraction complete.")
    except Exception as e:
        logger.critical(f"[NYUv2] Extraction FAILED: {e}")
        raise RuntimeError(f"Failed to extract NYUv2 archive: {e}") from e
    finally:
        # Clean up zip file to save disk space
        if zip_path.exists():
            zip_path.unlink()
            logger.info("[NYUv2] Cleaned up zip archive.")

    # --- Post-Download Verification ---
    # The zip may extract into a subdirectory called 'nyuv2'
    # Handle both cases: direct extraction and nested extraction
    nested = root_path / "nyuv2"
    if nested.is_dir() and (nested / "train").is_dir():
        # Move contents up one level
        import shutil
        for item in nested.iterdir():
            dest = root_path / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(str(dest))
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        nested.rmdir()
        logger.info("[NYUv2] Flattened nested directory structure.")

    if not _verify_integrity(root_path):
        raise RuntimeError(
            f"[NYUv2] Post-download integrity check FAILED at {root_path}. "
            f"Expected {EXPECTED_COUNTS} images per split."
        )

    logger.info(f"[NYUv2] Dataset ready at {root_path}")
    return root_path


def _verify_integrity(root_path: Path) -> bool:
    """
    Verify all expected files exist with correct counts.

    Checks:
    1. train/ and val/ directories exist
    2. Each split has image/, label/, depth/, normal/ subdirectories
    3. File counts match expected values (795 train, 654 val)
    """
    for split, expected_count in EXPECTED_COUNTS.items():
        split_dir = root_path / split

        if not split_dir.is_dir():
            logger.debug(f"[NYUv2] Missing split directory: {split_dir}")
            return False

        for modality in ["image", "label", "depth", "normal"]:
            mod_dir = split_dir / modality
            if not mod_dir.is_dir():
                logger.debug(f"[NYUv2] Missing modality directory: {mod_dir}")
                return False

            npy_files = list(mod_dir.glob("*.npy"))
            if len(npy_files) < expected_count:
                logger.debug(
                    f"[NYUv2] Incomplete {split}/{modality}: "
                    f"found {len(npy_files)}, expected {expected_count}"
                )
                return False

    return True


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    target = sys.argv[1] if len(sys.argv) > 1 else "data/nyuv2"
    download_nyuv2(root=target)
