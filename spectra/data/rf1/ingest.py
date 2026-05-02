"""
spectra/data/rf1/ingest.py
--------------------------
River Flow RF1 download and materialization pipeline.

This pipeline downloads the public Mulan benchmark ARFF files, maps the
benchmark's ``train`` and ``test`` splits onto SPECTRA's ``train`` and ``val``
convention, computes train-set normalization statistics, and writes a compact
file-based dataset without LMDB.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import arff

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "datasets" / "rf1"
STAGING_DIR = _PROJECT_ROOT / "data" / "staging_rf1"
CLEANUP_STAGING = True

DOWNLOADS = {
    "train": "https://sourceforge.net/projects/mulan/files/datasets/multi-target%20regression%20datasets/rf1-train.arff/download",
    "val": "https://sourceforge.net/projects/mulan/files/datasets/multi-target%20regression%20datasets/rf1-test.arff/download",
}

DEFAULT_FEATURE_NAMES = [
    f"site_{site_idx + 1}_lag_{lag_hours}h"
    for lag_hours in (0, 6, 12, 18, 24, 36, 48, 60)
    for site_idx in range(8)
]
DEFAULT_TARGET_NAMES = [f"flow_site_{site_idx + 1}_plus_48h" for site_idx in range(8)]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("spectra.rf1.ingest")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, destination)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _load_arff_matrix(path: Path) -> Tuple[np.ndarray, List[str]]:
    records, meta = arff.loadarff(path)
    column_names = list(meta.names())
    columns = [np.asarray(records[name], dtype=np.float32) for name in column_names]
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix, column_names


def _sanitize_names(source_names: List[str], fallback_names: List[str]) -> List[str]:
    if len(source_names) != len(fallback_names):
        return list(fallback_names)

    cleaned = []
    for idx, raw_name in enumerate(source_names):
        candidate = str(raw_name).strip().replace(" ", "_").replace("-", "_").lower()
        cleaned.append(candidate or fallback_names[idx])
    return cleaned


def _write_split(split_name: str, features: np.ndarray, targets: np.ndarray) -> None:
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_dir / "data.npz", features=features, targets=targets)


def _compute_imputation_values(train_features: np.ndarray) -> np.ndarray:
    column_means = np.nanmean(train_features, axis=0).astype(np.float32)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0).astype(np.float32)
    return column_means


def _impute_missing(features: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    filled = features.astype(np.float32, copy=True)
    nan_mask = np.isnan(filled)
    if nan_mask.any():
        row_idx, col_idx = np.where(nan_mask)
        filled[row_idx, col_idx] = fill_values[col_idx]
    return filled


def run_pipeline(force: bool = False, keep_staging: bool = False) -> None:
    expected_files = [
        OUTPUT_DIR / "metadata.json",
        OUTPUT_DIR / "train" / "data.npz",
        OUTPUT_DIR / "val" / "data.npz",
    ]
    if not force and all(path.exists() for path in expected_files):
        logger.info("RF1 data already exists. Use --force to regenerate.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    matrices: Dict[str, np.ndarray] = {}
    raw_column_names: List[str] | None = None

    for split_name, url in DOWNLOADS.items():
        staged_path = STAGING_DIR / f"{split_name}.arff"
        _download(url, staged_path)
        matrix, column_names = _load_arff_matrix(staged_path)
        matrices[split_name] = matrix
        raw_column_names = column_names
        logger.info("Loaded %s split with shape %s", split_name, tuple(matrix.shape))

    if raw_column_names is None:
        raise RuntimeError("RF1 download produced no columns.")

    if len(raw_column_names) < 72:
        raise ValueError(
            f"RF1 schema mismatch: expected at least 72 columns, found {len(raw_column_names)}"
        )

    feature_names = _sanitize_names(raw_column_names[:-8], DEFAULT_FEATURE_NAMES)
    target_names = list(DEFAULT_TARGET_NAMES)

    train_matrix = matrices["train"]
    val_matrix = matrices["val"]

    train_features = train_matrix[:, :-8].astype(np.float32, copy=False)
    train_targets = train_matrix[:, -8:].astype(np.float32, copy=False)
    val_features = val_matrix[:, :-8].astype(np.float32, copy=False)
    val_targets = val_matrix[:, -8:].astype(np.float32, copy=False)

    imputation_values = _compute_imputation_values(train_features)
    train_features = _impute_missing(train_features, imputation_values)
    val_features = _impute_missing(val_features, imputation_values)

    input_mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std[input_std < 1e-6] = 1.0

    _write_split("train", train_features, train_targets)
    _write_split("val", val_features, val_targets)

    metadata = {
        "dataset_name": "rf1",
        "benchmark": "rf1",
        "source": {
            "repository": "Mulan multi-target regression datasets",
            "train_url": DOWNLOADS["train"],
            "val_url": DOWNLOADS["val"],
        },
        "feature_names": feature_names,
        "target_names": target_names,
        "input_dim": int(train_features.shape[1]),
        "num_targets": int(train_targets.shape[1]),
        "train_size": int(train_features.shape[0]),
        "val_size": int(val_features.shape[0]),
        "input_stats": {
            "mean": input_mean.tolist(),
            "std": input_std.tolist(),
        },
        "imputation": {
            "strategy": "train_column_mean",
            "values": imputation_values.tolist(),
        },
    }

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info(
        "RF1 materialization complete: train=%d val=%d input_dim=%d targets=%d",
        metadata["train_size"],
        metadata["val_size"],
        metadata["input_dim"],
        metadata["num_targets"],
    )

    if keep_staging or not CLEANUP_STAGING:
        logger.info("Staging kept at %s", STAGING_DIR)
    elif STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
        logger.info("Staging removed: %s", STAGING_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPECTRA RF1 data generation")
    parser.add_argument("--force", action="store_true", help="Regenerate even if data exists")
    parser.add_argument("--keep-staging", action="store_true", help="Keep staged ARFF downloads")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SPECTRA RF1 DATA GENERATION")
    logger.info("=" * 60)
    run_pipeline(force=args.force, keep_staging=args.keep_staging)


if __name__ == "__main__":
    main()
