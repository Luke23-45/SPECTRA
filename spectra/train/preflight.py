"""
spectra/train/preflight.py
--------------------------
NASA-Tier Config Verification.
Validates everything before launching the runner to save compute.
"""

from typing import List
from pathlib import Path
import torch
import logging
from omegaconf import DictConfig

logger = logging.getLogger("spectra.preflight")

def preflight_check(cfg: DictConfig, output_dir: Path) -> None:
    """
    Validate all training prerequisites BEFORE allocating GPU compute.
    Principle: Fail fast and informatively.
    """
    errors: List[str] = []

    # 1. NYUv2 LMDB must exist before training
    dataset_name = cfg.get("dataset_name", cfg.get("dataset", {}).get("name", ""))
    if dataset_name == "nyuv2":
        root = cfg.get("root") or cfg.get("dataset", {}).get("root")
        if not root:
             errors.append("NYUv2 dataset selected but 'root' directory not defined.")
        else:
            lmdb_train = Path(root) / "train" / "data.lmdb"
            lmdb_val   = Path(root) / "val"   / "data.lmdb"
            if not lmdb_train.exists():
                errors.append(f"NYUv2 train LMDB missing: {lmdb_train}\n  → Run: python scripts/materialize_nyuv2.py")
            if not lmdb_val.exists():
                errors.append(f"NYUv2 val LMDB missing: {lmdb_val}\n  → Run: python scripts/materialize_nyuv2.py")

    # 2. Tasks must be defined
    tasks = list(cfg.get("tasks", []))
    if not tasks:
        errors.append("cfg.tasks is empty — no tasks configured. Check your dataset config.")

    # 3. Method name must be valid
    valid_methods = {"bpgs", "bpgs_alb", "kendall", "uwso", "pcgrad", "ntkmtl", "static"}
    method_name = cfg.get("method_name") or cfg.get("method", {}).get("name", "unknown")
    if method_name not in valid_methods:
        errors.append(f"Unknown method: '{method_name}'. Valid: {sorted(valid_methods)}")

    # 4. Loss names must be registered
    valid_losses = {"mse", "l1", "bce", "cross_entropy", "cosine", "masked_l1", "cosine_dense"}
    for task_cfg in tasks:
        loss_name = task_cfg.get("loss", "mse")
        if loss_name not in valid_losses:
            errors.append(f"Task '{task_cfg.get('name', '?')}': unknown loss '{loss_name}'. Valid: {sorted(valid_losses)}")

    # 5. Resume checkpoint exists if specified
    ckpt_path = cfg.get("resume_from", None)
    if ckpt_path and not Path(ckpt_path).exists():
        errors.append(f"Resume checkpoint not found: {ckpt_path}\n  → Check the path or remove 'resume_from' from config.")

    # 6. GPU memory sanity (warn only, do not block)
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size  = cfg.train.get("batch_size", 8)
        if benchmark == "nyuv2" and batch_size > 8 and free_mem_gb < 16.0:
            logger.warning(
                f"[PreFlight] batch_size={batch_size} on {free_mem_gb:.1f}GB GPU. "
                f"NYUv2/SegNet may OOM. Consider batch_size<=8 or train.accumulate_grad_batches=2."
            )
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logger.info(f"[PreFlight] Multi-GPU detected: {num_gpus} GPUs. Using DDP strategy.")

    # --- Report and Exit on Failures ---
    if errors:
        logger.error("[PreFlight] FAILED with the following errors:")
        for i, e in enumerate(errors, 1):
            logger.error(f"  [{i}] {e}")
        raise SystemExit(f"\n\nPre-flight check FAILED ({len(errors)} error(s)). Fix all errors above before training.")

    logger.info("[PreFlight] All systems nominal. GO for training.")
