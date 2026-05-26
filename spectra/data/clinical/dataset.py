"""
Clinical ICU data loader.

The loader reads prepared LMDB episodes, validates the expected 28-channel
schema, builds sliding windows, derives outcome and phase targets, and provides
optional training-time sensor perturbations.
"""

from __future__ import annotations

import os
import sys
import json
import lmdb
import logging
import functools
import numpy as np
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from torch.utils.data import Dataset, default_collate, Sampler, WeightedRandomSampler
from huggingface_hub import snapshot_download
from tqdm import tqdm
from spectra.engine.distributed import get_rank

logger = logging.getLogger("spectra.data.clinical")
logger.setLevel(logging.INFO)

EXPECTED_CHANNELS = 28
PHASE_STABLE = 0
PHASE_PRESHOCK = 1
PHASE_SHOCK = 2

def _is_outcome_positive_window(labels_window: np.ndarray, history_len: int) -> bool:
    """Outcome-positive iff sepsis appears in the prediction horizon."""
    fut_labels = labels_window[history_len:]
    return bool((fut_labels > 0.5).any())

# This order must match the prepared clinical dataset metadata.
CANONICAL_COLUMNS = [
    'HR', 'O2Sat', 'SBP', 'DBP', 'MAP', 'Resp', 'Temp',
    'Lactate', 'Creatinine', 'Bilirubin', 'Platelets', 'WBC',
    'pH', 'HCO3', 'BUN', 'Glucose', 'Hgb', 'Potassium',
    'Magnesium', 'Calcium', 'Chloride', 'FiO2',
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

COLUMN_GROUPS = {
    'hemodynamic': (0, 7),
    'labs': (7, 18),
    'electrolytes': (18, 22),
    'static': (22, 28),
}


# 0. TIERED DATA ORCHESTRATOR


def ensure_data_ready(
    dataset_dir: str = "data/ready",
    hf_repo_id: Optional[str] = None, 
    force_download: bool = False
) -> None:
    """Ensure that prepared clinical LMDB files are available."""
    dataset_path = Path(dataset_dir)
    required_splits = ["train", "val"]
    
    if not force_download:
        all_valid = True
        for split in required_splits:
            lmdb_p = dataset_path / split / "data.lmdb"
            idx_p = dataset_path.parent / f"{split}_index.json"
            if not idx_p.exists():
                idx_p = dataset_path / f"{split}_index.json"

            if not (lmdb_p.exists() and idx_p.exists()):
                logger.warning(f"[Tier 0] Missing split '{split}' artifacts in {dataset_path}")
                all_valid = False
                break
        
        if all_valid:
            logger.info(f"[Tier 0] Valid local data found at '{dataset_path}'.")
            return

    if hf_repo_id:
        logger.info(f"[Tier 1] Attempting download from HF Hub: {hf_repo_id}...")
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=dataset_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info("[Tier 1] Download completed.")
            return
        except Exception as e:
            logger.warning(f"[Tier 1] HF download failed: {e}. Falling back to local build.")

    logger.info("[Tier 2] Building clinical data from raw sources...")
    try:
        from .dataset_quality import build_quality_dataset as run_build_pipeline
        
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting Clinical Ingestor...")
        run_build_pipeline(output_dir=str(dataset_path))
        
        logger.info("[Tier 2] Build complete. Data is ready.")
    except ImportError:
        logger.critical("[Tier 2] Build Pipeline script not found. Cannot generate data.")
        raise RuntimeError("FATAL: Data missing and Build Pipeline unavailable.")
    except Exception as e:
        logger.critical(f"[Tier 2] Build Pipeline Crashed: {e}")
        raise RuntimeError("FATAL: Could not acquire or build ICU Dataset. Check connectivity and permissions.")

# 1. CORE DATASET


class ICUTrajectoryDataset(Dataset):
    """LMDB-backed ICU time-series dataset with sliding-window indexing."""
    def __init__(
        self,
        dataset_dir: str = "data/ready",
        split: str = "train",
        history_len: int = 24,
        pred_len: int = 6,
        max_cache_size: int = 128,
        validate_schema: bool = True,
        subset_pct: float = 1.0
    ):
        super().__init__()
        
        self.split = split
        self.history_len = history_len
        self.pred_len = pred_len
        self.window_size = history_len + pred_len
        self.subset_pct = subset_pct
        
        self.root_path = Path(dataset_dir) / split
        self.lmdb_path = self.root_path / "data.lmdb"
        self.index_path = self.root_path.parent / f"{split}_index.json"
        
        if not self.index_path.exists():
             self.index_path = self.root_path / f"{split}_index.json"

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB Critical Failure: Not found at {self.lmdb_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index Critical Failure: Not found at {self.index_path}")

        logger.info(f"[{split.upper()}] Loading Index: {self.index_path}")
        try:
            with open(self.index_path, 'r') as f:
                full_index = json.load(f)
                self.episode_metadata = full_index["episodes"]
                self.metadata = full_index["metadata"]
                self.global_stats = self.metadata.get("stats", None)
                
                if subset_pct < 1.0:
                    n_total = len(self.episode_metadata)
                    n_subset = max(1, int(n_total * subset_pct))
                    logger.info(f"[{split.upper()}] Piloting Mode: Subsetting to {subset_pct*100:.1f}% ({n_subset}/{n_total} episodes)")
                    self.episode_metadata = self.episode_metadata[:n_subset]
        except Exception as e:
            raise RuntimeError(f"Corrupted Index JSON: {e}")

        if validate_schema:
            ts_cols = self.metadata.get("ts_columns") or self.metadata.get("columns", [])
            
            if len(ts_cols) == 0:
                logger.warning("No column metadata found. Using configured clinical column order.")
                ts_cols = CANONICAL_COLUMNS
            elif len(ts_cols) != EXPECTED_CHANNELS:
                logger.error(f"SCHEMA MISMATCH! Expected {EXPECTED_CHANNELS}, Found {len(ts_cols)}")
                if len(ts_cols) < 20: 
                    raise ValueError(f"Dataset schema outdated ({len(ts_cols)} cols). Rebuild required.")
                logger.warning("Channel count mismatch. Using configured clinical column order.")
                ts_cols = CANONICAL_COLUMNS
            
            self.ts_columns = [
                'Bilirubin' if c == 'Bilirubin_total' else c 
                for c in ts_cols
            ]
            
            for i, (data_col, canonical_col) in enumerate(zip(self.ts_columns, CANONICAL_COLUMNS)):
                if data_col != canonical_col:
                    logger.warning(f"Column order mismatch at idx {i}: Data='{data_col}' vs Canon='{canonical_col}'")


        self.chunks_per_episode = []
        valid_episodes = 0
        
        for ep in self.episode_metadata:
            t_len = ep["length"]
            n_chunks = max(0, t_len - self.window_size + 1)
            self.chunks_per_episode.append(n_chunks)
            if n_chunks > 0:
                valid_episodes += 1

        self.cumulative_chunks = np.cumsum(self.chunks_per_episode)
        self.total_chunks = int(self.cumulative_chunks[-1]) if len(self.cumulative_chunks) > 0 else 0
        
        self._lmdb_env = None
        self._parent_pid = os.getpid()
        self.max_cache_size = max_cache_size

        logger.info(f"[{split.upper()}] Initialized. Windows: {self.total_chunks:,} | Episodes: {valid_episodes:,}")

    def __len__(self):
        return self.total_chunks

    def _init_lmdb(self):
        """Open an LMDB handle for the current process."""
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
                subdir=False
            )

    def _read_bytes(self, key: str) -> bytes:
        """Raw byte fetcher."""
        self._init_lmdb()
        with self._lmdb_env.begin(write=False) as txn:
            data = txn.get(key.encode('ascii'))
            if data is None:
                raise KeyError(f"LMDB Key failure: {key}. Index desynchronization detected.")
            return data

    def close(self):
        """Close the current LMDB handle."""
        if self._lmdb_env is not None:
            self._lmdb_env.close()
            self._lmdb_env = None
            logger.info(f"[{self.split.upper()}] LMDB Environment closed.")

    @functools.lru_cache(maxsize=512)
    def _fetch_numpy(self, key: str, dtype_str: str, shape: Tuple[int, ...]) -> np.ndarray:
        """Fetch and deserialize a NumPy array from LMDB."""
        raw = self._read_bytes(key)
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape).copy()

    def _get_phase_label(self, labels_window: np.ndarray) -> int:
        """Derive the phase label from observation and prediction windows."""
        obs_labels = labels_window[:self.history_len]
        fut_labels = labels_window[self.history_len:]
        
        if (obs_labels > 0.5).any():
            return PHASE_SHOCK
        elif (fut_labels > 0.5).any():
            return PHASE_PRESHOCK
        else:
            return PHASE_STABLE

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} out of bounds (Size: {self.total_chunks})")

        ep_idx = np.searchsorted(self.cumulative_chunks, idx, side='right')
        chunk_start_global = 0 if ep_idx == 0 else self.cumulative_chunks[ep_idx - 1]
        local_t_start = int(idx - chunk_start_global)

        ep_meta = self.episode_metadata[ep_idx]
        modalities = ep_meta.get("modalities", {})
        
        v_meta = modalities.get("vitals")
        if v_meta is None:
            raise KeyError(f"Episode {ep_meta.get('episode_id', ep_idx)} missing 'vitals'.")
        
        full_vitals = self._fetch_numpy(
            v_meta["key"], 
            v_meta.get("dtype", "float32"),
            tuple(v_meta["shape"])
        )
        
        if full_vitals.shape[1] != EXPECTED_CHANNELS:
            raise ValueError(
                f"Vitals channel mismatch: got {full_vitals.shape[1]}, expected {EXPECTED_CHANNELS}."
            )
        
        s_meta = modalities.get("static")
        if s_meta is not None:
            full_static = self._fetch_numpy(
                s_meta["key"], 
                s_meta.get("dtype", "float32"),
                tuple(s_meta["shape"])
            )
        else:
            static_start, static_end = COLUMN_GROUPS['static']
            full_static = full_vitals[0, static_start:static_end].copy()
        
        l_meta = modalities.get("labels")
        if l_meta is None:
            raise KeyError(f"Episode {ep_meta.get('episode_id', ep_idx)} missing 'labels'.")
        
        full_labels = self._fetch_numpy(
            l_meta["key"],
            l_meta.get("dtype", "float32"),
            tuple(l_meta["shape"])
        )

        m_meta = modalities.get("masks")
        if m_meta is None:
            full_masks = np.ones_like(full_vitals)
        else:
            full_masks = self._fetch_numpy(
                m_meta["key"], 
                m_meta.get("dtype", "float32"),
                tuple(m_meta["shape"])
            )

        t_end = local_t_start + self.window_size
        
        if t_end > len(full_vitals):
            raise ValueError(f"Window overrun for episode {ep_meta['episode_id']}")

        vitals_win = full_vitals[local_t_start : t_end]
        labels_win = full_labels[local_t_start : t_end]
        masks_win = full_masks[local_t_start : t_end]

        obs_data = vitals_win[:self.history_len]
        fut_data = vitals_win[self.history_len:]
        
        obs_mask = masks_win[:self.history_len]
        fut_mask = masks_win[self.history_len:]
        
        is_terminal = (t_end >= len(full_vitals))
        is_truncated = (t_end < len(full_vitals))

        phase = self._get_phase_label(labels_win)
        outcome = float(_is_outcome_positive_window(labels_win, self.history_len))

        return {
            "input": torch.from_numpy(obs_data.copy()),  # [T, 28]
            "targets": {
                "outcome": torch.tensor(outcome, dtype=torch.float32),
                "phase": torch.tensor(phase, dtype=torch.long),
            },
            "meta": {
                "future_data":   torch.from_numpy(fut_data.copy()),  # [Pred, 28]
                "static_context": torch.from_numpy(full_static.copy()), # [Stat]
                "src_mask":       torch.from_numpy(obs_mask.copy()),    # [T, 28]
                "future_mask":    torch.from_numpy(fut_mask.copy()),    # [Pred, 28]
                "is_terminal":    torch.tensor(is_terminal, dtype=torch.bool),
                "is_truncated":   torch.tensor(is_truncated, dtype=torch.bool),
                "patient_id":     str(ep_meta.get("patient_id", "unknown"))
            }
        }

class ICUDataset(ICUTrajectoryDataset):
    """ICU training dataset with optional sensor noise and channel dropout."""
    def __init__(
        self,
        dataset_dir: str = "data/ready",
        split: str = "train",
        history_len: int = 24,
        pred_len: int = 6,
        augment_noise: float = 0.005,
        augment_mask_prob: float = 0.0,
        validate_schema: bool = True,
        subset_pct: float = 1.0
    ):
        super().__init__(
            dataset_dir=dataset_dir, 
            split=split, 
            history_len=history_len, 
            pred_len=pred_len,
            validate_schema=validate_schema,
            subset_pct=subset_pct
        )
        
        self.augment_noise = augment_noise
        self.augment_mask_prob = augment_mask_prob
        self.is_training = (split == "train")
        
        if self.is_training:
            logger.info(f"Augmentation Active: Noise={augment_noise}, MaskDrop={augment_mask_prob}")

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            sample = super().__getitem__(idx)
            
            if torch.isnan(sample["input"]).any() or torch.isnan(sample["meta"]["future_data"]).any():
                logger.debug(f"Dropped NaN sample at idx {idx}")
                return None # Collator will filter this out

            if self.is_training:
                if self.augment_noise > 0:
                    noise = torch.randn_like(sample["input"]) * self.augment_noise
                    sample["input"] += noise
                
                if self.augment_mask_prob > 0:
                    mask = torch.rand(sample["input"].shape[1]) > self.augment_mask_prob
                    mask_broadcast = mask.float()
                    sample["input"] *= mask_broadcast
                    
                    if "src_mask" in sample["meta"]:
                         sample["meta"]["src_mask"] *= mask_broadcast

            return sample

        except Exception as e:
            logger.error(f"FATAL Load Error at idx {idx}: {e}", exc_info=False)
            return None

def robust_collate_fn(batch: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
    """Collate valid samples and drop samples rejected by the dataset."""
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        logger.warning("Empty Batch detected in Collate! (All samples failed robustness check)")
        return {}
    
    return default_collate(valid_batch)

class StatefulWeightedSampler(Sampler):
    """Weighted sampler with resumable epoch state."""
    def __init__(self, weights, num_samples, replacement=True, seed=42):
        # torch.utils.data.Sampler does not accept a positional argument on
        # newer PyTorch releases; use the version-safe no-arg constructor.
        super().__init__()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.consumed = 0
        self.rank = get_rank()
        self.indices = None

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Called by Trainer at start of epoch."""
        if epoch != self.epoch:
            self.consumed = 0
            self.epoch = epoch
            self.indices = None
        else:
            self.epoch = epoch
            pass

    def __iter__(self):
        if self.indices is None:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank * 1000000)
            
            self.indices = torch.multinomial(
                self.weights, 
                self.num_samples, 
                self.replacement, 
                generator=g
            )
        
        for i in range(self.consumed, self.num_samples):
            self.consumed += 1
            yield int(self.indices[i])
            
        self.indices = None
        self.consumed = 0

    def state_dict(self):
        """Return sampler state without storing sampled indices."""
        return {
            "consumed": self.consumed, 
            "epoch": self.epoch,
            "seed": self.seed
        }

    def load_state_dict(self, state_dict):
        self.consumed = state_dict.get("consumed", 0)
        self.epoch = state_dict.get("epoch", 0)
        self.seed = state_dict.get("seed", self.seed)
        self.indices = None
        logger.info(f"[Sampler] DDP-Link Restored: Epoch {self.epoch}, Consumed {self.consumed}")

def create_sepsis_aware_sampler(
    dataset: ICUTrajectoryDataset,
    sepsis_boost_factor: float = 10.0,
    max_samples: int = 100000,
    seed: int = 42,
    target: str = "outcome",
) -> StatefulWeightedSampler:
    """Create a weighted sampler that oversamples sepsis-positive windows."""
    n_samples = len(dataset)
    epoch_samples = min(n_samples, int(max_samples)) if max_samples is not None else n_samples
    if epoch_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    weights = torch.ones(n_samples)
    
    rank = get_rank()
    subset_str = getattr(dataset, "subset_pct", 1.0)
    if target not in {"outcome", "phase"}:
        raise ValueError(
            f"Unknown sampler target '{target}'. Valid: ['outcome', 'phase']"
        )

    index_name = f"{dataset.split}_sepsis_index_{target}_sub{subset_str}.npy"
    index_path = dataset.root_path / index_name
    
    if not index_path.exists() and rank != 0:
        logger.info(f"[Sampler] Rank {rank} waiting for Rank 0 to build index...")
        for _ in range(120): # 10 minute timeout
            if index_path.exists(): break
            time.sleep(5)
            
    if index_path.exists():
        try:
            logger.info(f"[Sampler] Loading cached Sepsis Index: {index_path}")
            is_sepsis = np.load(index_path)
            if len(is_sepsis) != n_samples:
                if rank == 0:
                    logger.warning("[Sampler] Index size mismatch! Rebuilding...")
                    index_path.unlink()
                else:
                    time.sleep(10)
                return create_sepsis_aware_sampler(
                    dataset, sepsis_boost_factor, max_samples, seed, target
                )
        except Exception as e:
            if rank == 0:
                logger.warning(f"[Sampler] Corrupt index detected, rebuilding: {e}")
                if index_path.exists(): index_path.unlink()
            else:
                time.sleep(10)
            return create_sepsis_aware_sampler(
                dataset, sepsis_boost_factor, max_samples, seed, target
            )
    else:
        logger.info(f"[Sampler] Rank {rank} building Global Sepsis Index (100% Coverage, N={n_samples:,})...")
        is_sepsis = np.zeros(n_samples, dtype=bool)
        
        dataset._init_lmdb()
        
        global_ptr = 0
        
        for ep_idx in tqdm(range(len(dataset.episode_metadata)), desc="Indexing Sepsis"):
            ep_meta = dataset.episode_metadata[ep_idx]
            n_chunks = dataset.chunks_per_episode[ep_idx]
            
            if n_chunks <= 0:
                continue
                
            l_meta = ep_meta["modalities"]["labels"]
            labels = dataset._fetch_numpy(
                l_meta["key"], 
                l_meta.get("dtype", "float32"),
                tuple(l_meta["shape"])
            )
            
            for local_idx in range(n_chunks):
                window = labels[local_idx : local_idx + dataset.window_size]
                if target == "phase":
                    is_positive = dataset._get_phase_label(window) > 0
                else:
                    is_positive = _is_outcome_positive_window(window, dataset.history_len)

                if is_positive:
                    is_sepsis[global_ptr + local_idx] = True
            
            global_ptr += n_chunks
            
        if rank == 0:
            try:
                temp_path = index_path.with_suffix(".tmp.npy")
                np.save(temp_path, is_sepsis)
                temp_path.replace(index_path)
                logger.info(f"[Sampler] Sepsis Index saved atomically to {index_path}")
            except Exception as e:
                logger.warning(f"[Sampler] Could not save Sepsis Index: {e}")
            
    weights[is_sepsis] = sepsis_boost_factor
    sepsis_count = int(is_sepsis.sum())
    rate = sepsis_count / n_samples
    
    logger.info(f"[Sampler] Coverage: 100% | Sepsis Detected: {sepsis_count:,} | Rate: {rate*100:.2f}% | Boost factor: {sepsis_boost_factor}x")
    
    sampler = StatefulWeightedSampler(
        weights=weights,
        num_samples=epoch_samples,
        replacement=True,
        seed=seed
    )
    logger.info(
        f"[Sampler] Epoch sample cap: {epoch_samples:,}/{n_samples:,} windows "
        f"({epoch_samples / n_samples:.2%} of dataset)"
    )
    
    return sampler
