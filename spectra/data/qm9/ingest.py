"""
QM9 tabular preparation with deterministic string-derived molecular features.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "datasets" / "qm9"
STAGING_DIR = _PROJECT_ROOT / "data" / "staging_qm9"
CLEANUP_STAGING = True
SPLIT_SEED = 42
TRAIN_FRAC = 0.9

DOWNLOAD_URLS = [
    "https://huggingface.co/datasets/n0w0f/qm9-csv/resolve/main/qm9_dataset.csv?download=true",
    "https://huggingface.co/datasets/n0w0f/qm9-csv/resolve/main/qm9_dataset.csv",
]

TARGET_COLUMNS = [
    ("dipole_moment", "mu"),
    ("polarizability", "alpha"),
    ("homo", "homo"),
    ("lumo", "lumo"),
    ("gap", "gap"),
    ("r2", "r2"),
    ("zero_point_energy", "zpve"),
    ("u0", "u0"),
    ("u298", "u298"),
    ("h298", "h298"),
    ("g298", "g298"),
    ("heat_capacity", "cv"),
]

FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")
logger = logging.getLogger("spectra.qm9.ingest")
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


def _parse_formula(inchi: str) -> Dict[str, int]:
    if not isinstance(inchi, str) or "/" not in inchi:
        return {}
    formula = inchi.split("/", 1)[1].split("/", 1)[0]
    counts = {}
    for atom, raw_count in FORMULA_PATTERN.findall(formula):
        counts[atom] = counts.get(atom, 0) + int(raw_count or "1")
    return counts


def _smiles_features(smiles: str) -> Dict[str, float]:
    smiles = smiles or ""
    digit_count = sum(ch.isdigit() for ch in smiles)
    lower_count = sum(ch.islower() for ch in smiles)
    return {
        "smiles_length": float(len(smiles)),
        "branch_count": float(smiles.count("(") + smiles.count(")")),
        "double_bond_count": float(smiles.count("=")),
        "triple_bond_count": float(smiles.count("#")),
        "ring_marker_count": float(digit_count),
        "ring_count_estimate": float(digit_count / 2.0),
        "aromatic_char_count": float(lower_count),
        "chirality_count": float(smiles.count("@")),
        "charge_plus_count": float(smiles.count("+")),
        "charge_minus_count": float(smiles.count("-")),
        "slash_count": float(smiles.count("/") + smiles.count("\\")),
        "bracket_count": float(smiles.count("[") + smiles.count("]")),
    }


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for smiles, inchi in zip(df["smiles"].astype(str), df["inchi"].astype(str)):
        formula = _parse_formula(inchi)
        heavy_atoms = formula.get("C", 0) + formula.get("N", 0) + formula.get("O", 0) + formula.get("F", 0)
        total_atoms = heavy_atoms + formula.get("H", 0)
        feat = {
            "count_C": float(formula.get("C", 0)),
            "count_H": float(formula.get("H", 0)),
            "count_N": float(formula.get("N", 0)),
            "count_O": float(formula.get("O", 0)),
            "count_F": float(formula.get("F", 0)),
            "heavy_atom_count": float(heavy_atoms),
            "total_atom_count": float(total_atoms),
            "hydrogen_to_heavy_ratio": float(formula.get("H", 0) / max(heavy_atoms, 1)),
            "hetero_atom_count": float(formula.get("N", 0) + formula.get("O", 0) + formula.get("F", 0)),
            "unique_atom_types": float(sum(1 for atom in ["C", "H", "N", "O", "F"] if formula.get(atom, 0) > 0)),
        }
        feat.update(_smiles_features(smiles))
        feat["smiles_atom_C"] = float(smiles.count("C") + smiles.count("c"))
        feat["smiles_atom_N"] = float(smiles.count("N") + smiles.count("n"))
        feat["smiles_atom_O"] = float(smiles.count("O") + smiles.count("o"))
        feat["smiles_atom_F"] = float(smiles.count("F"))
        rows.append(feat)
    return pd.DataFrame(rows)


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
        logger.info("QM9 data already exists. Use --force to regenerate.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    staged_csv = STAGING_DIR / "qm9.csv"
    source_url = _download(DOWNLOAD_URLS, staged_csv)
    df = pd.read_csv(staged_csv)

    required_columns = {"smiles", "inchi"} | {column for column, _ in TARGET_COLUMNS}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"QM9 CSV missing required columns: {missing}")

    feature_df = _build_feature_frame(df)
    features = feature_df.to_numpy(dtype=np.float32, copy=False)
    target_names = [alias for _, alias in TARGET_COLUMNS]
    targets = df[[column for column, _ in TARGET_COLUMNS]].to_numpy(dtype=np.float32, copy=False)

    if not np.isfinite(features).all():
        raise ValueError("QM9 featurization produced non-finite inputs.")
    if not np.isfinite(targets).all():
        raise ValueError("QM9 targets contain non-finite values.")

    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(features))
    train_count = int(len(indices) * TRAIN_FRAC)
    train_idx = np.sort(indices[:train_count])
    val_idx = np.sort(indices[train_count:])

    train_features = features[train_idx]
    train_targets = targets[train_idx]
    val_features = features[val_idx]
    val_targets = targets[val_idx]

    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0

    _write_split("train", train_features, train_targets)
    _write_split("val", val_features, val_targets)

    metadata = {
        "dataset_name": "qm9",
        "benchmark": "qm9",
        "source": {
            "repository": "n0w0f/qm9-csv Hugging Face mirror",
            "url": source_url,
            "original_dataset": "QM9 (Ramakrishnan et al., 2014)",
        },
        "feature_names": feature_df.columns.tolist(),
        "target_names": target_names,
        "source_target_columns": [column for column, _ in TARGET_COLUMNS],
        "input_dim": int(train_features.shape[1]),
        "num_targets": int(train_targets.shape[1]),
        "train_size": int(train_features.shape[0]),
        "val_size": int(val_features.shape[0]),
        "split": {
            "strategy": "seeded_random_split",
            "seed": SPLIT_SEED,
            "train_fraction": TRAIN_FRAC,
        },
        "input_stats": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info(
        "QM9 materialization complete: train=%d val=%d input_dim=%d targets=%d",
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
    parser = argparse.ArgumentParser(description="SPECTRA QM9 data generation")
    parser.add_argument("--force", action="store_true", help="Regenerate even if data exists")
    parser.add_argument("--keep-staging", action="store_true", help="Keep staged downloads")
    args = parser.parse_args()
    logger.info("=" * 60)
    logger.info("SPECTRA QM9 DATA GENERATION")
    logger.info("=" * 60)
    run_pipeline(force=args.force, keep_staging=args.keep_staging)


if __name__ == "__main__":
    main()
