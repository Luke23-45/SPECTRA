"""
Prepared QM9 multi-target regression dataset.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("spectra.data.qm9")


def resolve_qm9_root(root: str | Path) -> Path:
    root_path = Path(root)
    candidates = [root_path, root_path / "data"]
    for candidate in candidates:
        if (candidate / "metadata.json").exists() and (
            (candidate / "train" / "data.npz").exists()
            or (candidate / "val" / "data.npz").exists()
        ):
            return candidate
    return root_path


class QM9Dataset(Dataset):
    def __init__(
        self,
        root: str = "datasets/qm9",
        split: str = "train",
        normalize_inputs: bool = True,
        subset_pct: float = 1.0,
        subset_seed: int = 42,
    ):
        super().__init__()
        self.root = resolve_qm9_root(root)
        self.split = "val" if split in {"validation", "val", "test"} else "train"
        self.data_path = self.root / self.split / "data.npz"
        self.metadata_path = self.root / "metadata.json"

        if not self.data_path.exists():
            raise FileNotFoundError(f"[QM9] Split data missing at: {self.data_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"[QM9] Metadata missing at: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        packed = np.load(self.data_path, allow_pickle=False)
        self.features = packed["features"].astype(np.float32, copy=False)
        self.targets_matrix = packed["targets"].astype(np.float32, copy=False)

        self.feature_names = list(self.metadata["feature_names"])
        self.target_names = list(self.metadata["target_names"])
        self.input_dim = int(self.features.shape[1])
        self.num_targets = int(self.targets_matrix.shape[1])

        stats = self.metadata.get("input_stats", {})
        mean = np.asarray(stats.get("mean", [0.0] * self.input_dim), dtype=np.float32)
        std = np.asarray(stats.get("std", [1.0] * self.input_dim), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

        if normalize_inputs:
            self.features = (self.features - mean) / std

        all_indices = list(range(len(self.features)))
        if 0.0 < subset_pct < 1.0:
            rng = random.Random(subset_seed)
            count = max(1, int(len(all_indices) * subset_pct))
            self.indices = sorted(rng.sample(all_indices, count))
        else:
            self.indices = all_indices

        logger.info(
            "[QM9-%s] Initialized: %d samples, input_dim=%d, targets=%d, normalized=%s",
            self.split.upper(),
            len(self.indices),
            self.input_dim,
            self.num_targets,
            "ON" if normalize_inputs else "OFF",
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.indices[idx]
        features = torch.from_numpy(self.features[global_idx])
        target_row = torch.from_numpy(self.targets_matrix[global_idx])
        return {
            "input": features,
            "targets": {
                name: target_row[target_idx]
                for target_idx, name in enumerate(self.target_names)
            },
            "meta": {"sample_id": f"qm9_{self.split}_{global_idx}"},
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = torch.stack([item["input"] for item in batch])
        target_keys = batch[0]["targets"].keys()
        targets = {
            key: torch.stack([item["targets"][key] for item in batch])
            for key in target_keys
        }
        meta = {"sample_id": [item["meta"]["sample_id"] for item in batch]}
        return {"input": inputs, "targets": targets, "meta": meta}
