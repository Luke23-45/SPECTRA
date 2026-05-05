from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from studies.bpgs_study.common.paths import PROJECT_ROOT, resolve_study_output_root
from studies.bpgs_study.common.specs import StudySpec


def _get_system_info() -> Dict[str, str]:
    """Collect current system information."""
    import torch
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
        "pytorch_lightning_version": "",
        "numpy_version": "",
        "scipy_version": "",
    }
    try:
        import pytorch_lightning as pl
        info["pytorch_lightning_version"] = pl.__version__
    except ImportError:
        pass
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = str(torch.cuda.device_count())
    return info


def _get_git_info(repo_root: Path) -> Dict[str, Optional[str]]:
    """Extract git information from the repository."""
    git_info: Dict[str, Optional[str]] = {
        "git_sha": None,
        "git_branch": None,
        "git_dirty": None,
        "git_describe": None,
    }
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return git_info

    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(repo_root), timeout=5,
        )
        if result.returncode == 0:
            git_info["git_sha"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=str(repo_root), timeout=5,
        )
        if result.returncode == 0:
            git_info["git_branch"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(repo_root), timeout=5,
        )
        if result.returncode == 0:
            git_info["git_dirty"] = str(len(result.stdout.strip()) > 0)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, cwd=str(repo_root), timeout=5,
        )
        if result.returncode == 0:
            git_info["git_describe"] = result.stdout.strip()
    except Exception:
        pass

    return git_info


def _get_package_versions() -> Dict[str, str]:
    """Collect versions of key packages."""
    packages = [
        "torch", "pytorch_lightning", "numpy", "scipy", "pandas",
        "matplotlib", "seaborn", "hydra_core", "omegaconf", "wandb",
        "torchmetrics", "scikit-learn", "tqdm", "lmdb",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "NOT_INSTALLED"
    return versions


def build_study_manifest(
    study: StudySpec,
    study_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build a comprehensive reproducibility manifest for a study.

    Collects:
    - Study specification
    - System information
    - Git state
    - Package versions
    - Per-run metadata inventory
    - Config digest
    """
    if study_root is None:
        study_root = resolve_study_output_root(study.name)

    manifest: Dict[str, Any] = {
        "manifest_version": "2.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "study": {
            "name": study.name,
            "kind": study.kind,
            "group": study.group,
            "description": study.description,
            "notes": study.notes,
            "config_path": str(study.config_path),
            "requires_nyuv2_subsets": study.requires_nyuv2_subsets,
            "n_variants": len(study.variants),
            "variants": [
                {
                    "label": v.label,
                    "dataset": getattr(v, "dataset", None),
                    "method": getattr(v, "method", None),
                    "epochs": getattr(v, "epochs", None),
                    "seeds": list(v.seeds),
                }
                for v in study.variants
            ],
        },
        "system": _get_system_info(),
        "git": _get_git_info(PROJECT_ROOT),
        "packages": _get_package_versions(),
        "run_inventory": [],
    }

    # Inventory all run directories
    if study_root.exists():
        for run_dir in sorted(study_root.glob("*/*")):
            if not run_dir.is_dir():
                continue
            run_entry: Dict[str, Any] = {
                "path": str(run_dir),
                "variant": run_dir.parent.name,
                "dir_name": run_dir.name,
            }

            # Read metadata if available
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    run_entry["method"] = metadata.get("experiment", {}).get("method")
                    run_entry["dataset"] = metadata.get("experiment", {}).get("dataset")
                    run_entry["seed"] = metadata.get("experiment", {}).get("seed")
                    run_entry["run_id"] = metadata.get("experiment", {}).get("run_id")
                    run_entry["system"] = metadata.get("system")
                    run_entry["git"] = metadata.get("git")
                    run_entry["training_config"] = metadata.get("training")
                    run_entry["tasks"] = metadata.get("tasks")
                except Exception:
                    pass

            # Read run summary if available
            rs_path = run_dir / "run_summary.json"
            if rs_path.exists():
                try:
                    rs = json.loads(rs_path.read_text(encoding="utf-8"))
                    run_entry["stopped_epoch"] = rs.get("stopped_epoch")
                    run_entry["global_step"] = rs.get("global_step")
                    run_entry["elapsed_seconds"] = rs.get("elapsed_seconds")
                    run_entry["selection_metric"] = rs.get("selection_metric")
                    run_entry["selected_checkpoint"] = rs.get("selected_checkpoint")
                except Exception:
                    pass

            # Check for config
            config_path = run_dir / "config.yaml"
            run_entry["has_config"] = config_path.exists()

            # Check for checkpoints
            ckpt_dir = run_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                run_entry["n_checkpoints"] = len(ckpts)
                run_entry["checkpoint_files"] = [c.name for c in ckpts]
            else:
                run_entry["n_checkpoints"] = 0

            # Check for CSV logs
            csv_logs = list((run_dir / "csv_logs").glob("*/*.csv"))
            run_entry["n_csv_logs"] = len(csv_logs)

            manifest["run_inventory"].append(run_entry)

    # Compute completeness
    total_expected = sum(len(v.seeds) for v in study.variants)
    total_found = len(manifest["run_inventory"])
    manifest["completeness"] = {
        "expected_runs": total_expected,
        "found_runs": total_found,
        "complete": total_found >= total_expected,
        "missing_runs": max(0, total_expected - total_found),
    }

    # Config digest
    if study.config_path.exists():
        import hashlib
        config_bytes = study.config_path.read_bytes()
        manifest["config_digest"] = {
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "size_bytes": len(config_bytes),
        }

    return manifest


def build_cross_study_manifest(
    studies: List[StudySpec],
) -> Dict[str, Any]:
    """Build a master reproducibility manifest across all studies."""
    manifest: Dict[str, Any] = {
        "manifest_version": "2.0",
        "manifest_type": "cross_study",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_studies": len(studies),
        "system": _get_system_info(),
        "git": _get_git_info(PROJECT_ROOT),
        "packages": _get_package_versions(),
        "studies": [],
    }

    total_expected = 0
    total_found = 0

    for study in studies:
        study_manifest = build_study_manifest(study)
        manifest["studies"].append(study_manifest)
        completeness = study_manifest.get("completeness", {})
        total_expected += completeness.get("expected_runs", 0)
        total_found += completeness.get("found_runs", 0)

    manifest["overall_completeness"] = {
        "total_expected_runs": total_expected,
        "total_found_runs": total_found,
        "complete": total_found >= total_expected,
    }

    return manifest


def write_study_manifest(
    study: StudySpec,
    output_dir: Path,
) -> Path:
    """Write reproducibility manifest for a single study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_study_manifest(study)
    path = output_dir / "reproducibility_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def write_cross_study_manifest(
    studies: List[StudySpec],
    output_dir: Path,
) -> Path:
    """Write master reproducibility manifest across all studies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_cross_study_manifest(studies)
    path = output_dir / "master_reproducibility_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
