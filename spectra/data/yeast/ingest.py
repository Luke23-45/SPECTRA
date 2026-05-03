"""
spectra/data/yeast/ingest.py
----------------------------
Yeast Protein Localization multi-label data preparation.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import arff

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "datasets" / "yeast"
STAGING_DIR = _PROJECT_ROOT / "data" / "staging_yeast"
CLEANUP_STAGING = True
LABEL_COUNT = 14

DOWNLOADS = {
    "train": [
        "https://sourceforge.net/projects/meka/files/Datasets/Train-test%20Splits/yeast-train.arff/download",
    ],
    "val": [
        "https://sourceforge.net/projects/meka/files/Datasets/Train-test%20Splits/yeast-test.arff/download",
    ],
}

logger = logging.getLogger("spectra.yeast.ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _download(urls: List[str], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in urls:
        try:
            logger.info("Downloading %s -> %s", url, destination)
            with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return url
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
            logger.warning("Download failed from %s: %s", url, exc)
    raise RuntimeError(f"All download URLs failed for {destination.name}: {last_error}")


def _load_arff(path: Path) -> Tuple[np.ndarray, List[str]]:
    records, meta = arff.loadarff(path)
    column_names = list(meta.names())
    columns = []
    for name in column_names:
        col = np.asarray(records[name])
        if col.dtype.kind in {"S", "U", "O"}:
            decoded = np.array([float(x.decode("utf-8") if isinstance(x, bytes) else x) for x in col], dtype=np.float32)
            columns.append(decoded)
        else:
            columns.append(col.astype(np.float32))
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix, column_names


def _write_split(split_name: str, features: np.ndarray, targets: np.ndarray) -> None:
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_dir / "data.npz", features=features, targets=targets)


def run_pipeline(force: bool = False, keep_staging: bool = False) -> None:
    expected_files = [
        OUTPUT_DIR / "metadata.json",
        OUTPUT_DIR / "train" / "data.npz",
        OUTPUT_DIR / "val" / "data.npz",
    ]
    if not force and all(path.exists() for path in expected_files):
        logger.info("Yeast data already exists. Use --force to regenerate.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    source_urls = {}
    matrices = {}
    column_names = None
    for split_name, urls in DOWNLOADS.items():
        staged_path = STAGING_DIR / f"{split_name}.arff"
        source_urls[split_name] = _download(urls, staged_path)
        matrix, names = _load_arff(staged_path)
        matrices[split_name] = matrix
        column_names = names
        logger.info("Loaded %s split with shape %s", split_name, tuple(matrix.shape))

    if column_names is None:
        raise RuntimeError("Yeast ingestion produced no schema.")
    if len(column_names) <= LABEL_COUNT:
        raise ValueError(f"Yeast schema mismatch: expected > {LABEL_COUNT} columns, got {len(column_names)}")

    feature_names = [str(name) for name in column_names[:-LABEL_COUNT]]
    source_target_names = [str(name) for name in column_names[-LABEL_COUNT:]]
    target_names = [f"label_{idx + 1:02d}" for idx in range(LABEL_COUNT)]

    train_matrix = matrices["train"]
    val_matrix = matrices["val"]
    train_features = train_matrix[:, :-LABEL_COUNT].astype(np.float32, copy=False)
    train_targets = train_matrix[:, -LABEL_COUNT:].astype(np.float32, copy=False)
    val_features = val_matrix[:, :-LABEL_COUNT].astype(np.float32, copy=False)
    val_targets = val_matrix[:, -LABEL_COUNT:].astype(np.float32, copy=False)

    unique_targets = set(np.unique(train_targets).tolist()) | set(np.unique(val_targets).tolist())
    if not unique_targets.issubset({0.0, 1.0}):
        raise ValueError(f"Yeast labels are not binary: observed {sorted(unique_targets)}")

    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0

    _write_split("train", train_features, train_targets)
    _write_split("val", val_features, val_targets)

    metadata = {
        "dataset_name": "yeast",
        "benchmark": "yeast",
        "source": {
            "repository": "MEKA train-test splits",
            "train_url": source_urls["train"],
            "val_url": source_urls["val"],
            "format_note": "Mulan format with labels at the end of each row",
        },
        "feature_names": feature_names,
        "target_names": target_names,
        "source_target_names": source_target_names,
        "input_dim": int(train_features.shape[1]),
        "num_labels": int(train_targets.shape[1]),
        "train_size": int(train_features.shape[0]),
        "val_size": int(val_features.shape[0]),
        "input_stats": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info(
        "Yeast materialization complete: train=%d val=%d input_dim=%d labels=%d",
        metadata["train_size"],
        metadata["val_size"],
        metadata["input_dim"],
        metadata["num_labels"],
    )

    if keep_staging or not CLEANUP_STAGING:
        logger.info("Staging kept at %s", STAGING_DIR)
    elif STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
        logger.info("Staging removed: %s", STAGING_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPECTRA Yeast data generation")
    parser.add_argument("--force", action="store_true", help="Regenerate even if data exists")
    parser.add_argument("--keep-staging", action="store_true", help="Keep staged downloads")
    args = parser.parse_args()
    logger.info("=" * 60)
    logger.info("SPECTRA YEAST DATA GENERATION")
    logger.info("=" * 60)
    run_pipeline(force=args.force, keep_staging=args.keep_staging)


if __name__ == "__main__":
    main()
