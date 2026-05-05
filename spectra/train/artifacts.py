"""
spectra/train/artifacts.py
--------------------------
Stable artifact and resume-path utilities for long-running training jobs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import torch


def _original_cwd() -> Path:
    try:
        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def resolve_artifact_dir(cfg: DictConfig) -> Path:
    """Resolve the stable run artifact directory outside Hydra timestamp shells."""
    run_name = cfg.get("run_name", "unnamed_run")
    rel_path = cfg.get("output_dir", f"./outputs/{run_name}")
    return (_original_cwd() / Path(rel_path)).resolve()


def resolve_resume_checkpoint(cfg: DictConfig, artifact_dir: Path) -> Optional[Path]:
    """
    Resolve the checkpoint path for resume.

    Supported values:
      - null / empty: no resume
      - "auto": resume from `<artifact_dir>/checkpoints/last.ckpt` if present
      - explicit path: absolute or relative to the original working directory
    """
    raw_resume = cfg.get("resume_from", None)
    if raw_resume in (None, "", False):
        return None

    resume_str = str(raw_resume).strip()
    if not resume_str:
        return None

    if resume_str.lower() == "auto":
        candidate = artifact_dir / "checkpoints" / "last.ckpt"
        return candidate if candidate.exists() else None

    candidate = Path(resume_str)
    if not candidate.is_absolute():
        candidate = (_original_cwd() / candidate).resolve()
    return candidate


def stable_run_id(cfg: DictConfig) -> str:
    """Stable logger/run identifier safe for WandB resume."""
    run_name = str(cfg.get("run_name", "unnamed_run"))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")
    return safe or "unnamed-run"


def resolve_selection_config(cfg: DictConfig) -> Dict[str, str]:
    train_cfg = cfg.get("train", {})
    monitor = str(train_cfg.get("selection_metric", "val/total_loss"))
    mode = str(train_cfg.get("selection_mode", "min"))
    return {"metric": monitor, "mode": mode}


def _get_git_info() -> Dict[str, Any]:
    """Get git commit hash, branch, and dirty status."""
    git_info = {
        "commit_hash": "unknown",
        "branch": "unknown",
        "is_dirty": False
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_original_cwd()
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
        
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_original_cwd()
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=_original_cwd()
        )
        if result.returncode == 0:
            git_info["is_dirty"] = len(result.stdout.strip()) > 0
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return git_info


def _get_system_info() -> Dict[str, Any]:
    """Get system and environment information."""
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": (
            torch.backends.cuda.matmul.allow_tf32
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32")
            else None
        ),
        "tf32_cudnn": getattr(torch.backends.cudnn, "allow_tf32", None),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }


def save_experiment_config(cfg: DictConfig, artifact_dir: Path) -> Path:
    """
    Save the resolved configuration to artifact_dir/config.yaml.
    
    Args:
        cfg: The resolved Hydra configuration
        artifact_dir: The stable artifact directory
        
    Returns:
        Path to the saved config file
    """
    config_path = artifact_dir / "config.yaml"
    
    # Save as resolved YAML (all interpolations resolved)
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    
    return config_path


def generate_experiment_metadata(cfg: DictConfig, artifact_dir: Path) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for publication reproducibility.
    
    Args:
        cfg: The resolved Hydra configuration
        artifact_dir: The stable artifact directory
        
    Returns:
        Dictionary containing all metadata
    """
    train_cfg = cfg.get("train", {})
    resolved_resume = resolve_resume_checkpoint(cfg, artifact_dir)
    method_name = cfg.get("method_name") or cfg.get("method", {}).get("name", "unknown")
    dataset_name = cfg.get("dataset_name", "unknown")
    
    # Extract task information
    tasks = cfg.get("tasks", [])
    task_info = []
    for task in tasks:
        task_info.append({
            "name": task.get("name", "unknown"),
            "type": task.get("type", "unknown"),
            "loss": task.get("loss", "unknown"),
            "manifold": task.get("manifold", "unknown"),
            "weight": task.get("weight", 1.0)
        })
    
    # Extract dataset-specific info
    dataset_info = {
        "name": dataset_name,
        "augmentation": cfg.get("augmentation", False),
        "subset_pct": cfg.get("subset_pct", 1.0),
        "subset_seed": cfg.get("subset_seed", 42),
        "train_subset_file": cfg.get("train_subset_file", None),
        "train_subset_id": cfg.get("train_subset_id", None),
    }
    
    # Add dataset-specific fields
    if dataset_name == "nyuv2":
        dataset_info.update({
            "num_classes": cfg.get("num_classes", 13),
            "ignore_index": cfg.get("ignore_index", 255),
            "image_height": cfg.get("image_height", 288),
            "image_width": cfg.get("image_width", 384),
            "train_size": cfg.get("train_size", 0),
            "val_size": cfg.get("val_size", 0)
        })
    elif dataset_name == "rf1":
        dataset_info.update({
            "input_dim": cfg.get("input_dim", cfg.get("model", {}).get("input_dim", 0)),
            "num_targets": cfg.get("num_targets", 0),
            "normalize_inputs": cfg.get("normalize_inputs", True),
            "train_size": cfg.get("train_size", 0),
            "val_size": cfg.get("val_size", 0),
        })
    elif dataset_name == "yeast":
        dataset_info.update({
            "input_dim": cfg.get("input_dim", cfg.get("model", {}).get("input_dim", 0)),
            "num_labels": cfg.get("num_labels", 0),
            "normalize_inputs": cfg.get("normalize_inputs", True),
            "train_size": cfg.get("train_size", 0),
            "val_size": cfg.get("val_size", 0),
        })
    elif dataset_name == "qm9":
        dataset_info.update({
            "input_dim": cfg.get("input_dim", cfg.get("model", {}).get("input_dim", 0)),
            "num_targets": cfg.get("num_targets", 0),
            "normalize_inputs": cfg.get("normalize_inputs", True),
            "train_size": cfg.get("train_size", 0),
            "val_size": cfg.get("val_size", 0),
        })
    
    metadata = {
        "experiment": {
            "run_name": cfg.get("run_name", "unnamed_run"),
            "run_id": stable_run_id(cfg),
            "method": method_name,
            "dataset": dataset_name,
            "seed": cfg.get("seed", 42)
        },
        "system": _get_system_info(),
        "git": _get_git_info(),
        "dataset": dataset_info,
        "training": {
            "epochs": train_cfg.get("epochs", 0),
            "batch_size": train_cfg.get("batch_size", 0),
            "lr": train_cfg.get("lr", 0.0),
            "min_lr": train_cfg.get("min_lr", 0.0),
            "warmup_steps": train_cfg.get("warmup_steps", 0),
            "weight_decay": train_cfg.get("weight_decay", 0.0),
            "grad_clip": train_cfg.get("grad_clip", 0.0),
            "precision": train_cfg.get("precision", "32"),
            "num_workers": train_cfg.get("num_workers", 0),
            "deterministic": train_cfg.get("deterministic", False),
            "deterministic_warn_only": train_cfg.get("deterministic_warn_only", False),
            "early_stop": train_cfg.get("early_stop", False),
            "early_stop_patience": train_cfg.get("early_stop_patience", 0),
            "checkpoint_every_minutes": train_cfg.get("checkpoint_every_minutes", 0),
            "selection_metric": resolve_selection_config(cfg)["metric"],
            "selection_mode": resolve_selection_config(cfg)["mode"],
            "progress": OmegaConf.to_container(train_cfg.get("progress", {}), resolve=True),
        },
        "resume": {
            "requested": None if cfg.get("resume_from", None) in (None, "", False) else str(cfg.get("resume_from")),
            "resolved_checkpoint": str(resolved_resume) if resolved_resume is not None else None,
            "auto_resume_found": bool(resolved_resume) if str(cfg.get("resume_from", "")).strip().lower() == "auto" else None,
        },
        "tasks": task_info,
        "paths": {
            "artifact_dir": str(artifact_dir),
            "config": "config.yaml",
            "metadata": "metadata.json"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    return metadata


def save_experiment_metadata(cfg: DictConfig, artifact_dir: Path) -> Path:
    """
    Generate and save metadata.json to artifact_dir.
    
    Args:
        cfg: The resolved Hydra configuration
        artifact_dir: The stable artifact directory
        
    Returns:
        Path to the saved metadata file
    """
    metadata = generate_experiment_metadata(cfg, artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata_path


def _checkpoint_epoch_from_path(path: str | None) -> Optional[int]:
    if not path:
        return None
    match = re.search(r"ep(\d+)", str(path))
    if not match:
        return None
    return int(match.group(1))


def save_run_summary(
    cfg: DictConfig,
    artifact_dir: Path,
    summary: Dict[str, Any],
) -> Path:
    """
    Save a post-fit execution summary used by paper aggregation scripts.
    """
    selection = resolve_selection_config(cfg)
    checkpoint_registry = summary.get("checkpoint_registry", {})
    selected = checkpoint_registry.get(selection["metric"], {})

    normalized_summary = {
        "fit_started_at": summary.get("fit_started_at"),
        "fit_ended_at": summary.get("fit_ended_at"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "stopped_epoch": summary.get("stopped_epoch"),
        "global_step": summary.get("global_step"),
        "selection_metric": selection["metric"],
        "selection_mode": selection["mode"],
        "selected_checkpoint": {
            "path": selected.get("best_model_path"),
            "score": selected.get("best_model_score"),
            "epoch": _checkpoint_epoch_from_path(selected.get("best_model_path")),
        },
        "checkpoint_registry": checkpoint_registry,
    }

    summary_path = artifact_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized_summary, handle, indent=2, default=str)
    return summary_path
