"""
spectra/utils/progress.py
-------------------------
Progress interfaces for local terminals and log-heavy notebook runtimes.
"""

from __future__ import annotations

import math
import time
from typing import Any

from tqdm import tqdm
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback, TQDMProgressBar


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SOTAProgressBar(TQDMProgressBar):
    """
    Rich local-terminal progress view.
    Maps long metric keys to concise research shorthand.
    """

    def __init__(self, *args, **kwargs):
        if "leave" not in kwargs:
            kwargs["leave"] = True
        super().__init__(*args, **kwargs)

    def init_train_tqdm(self) -> tqdm:
        bar = super().init_train_tqdm()
        bar.bar_format = "{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = super().init_validation_tqdm()
        bar.bar_format = "{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return bar

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)

        mapping = {
            "loss": "L",
            "train/total_loss": "L",
            "health/backbone_grad_norm": "GN",
            "health/update_weight_ratio": "Ratio",
            "pcgrad/total_conflicts": "C",
            "spectral/hf_divorce_index": "D",
            "health/backbone_weight_norm": "WN",
            "val/total_loss": "vL",
            "train/AUC": "AUC",
            "train/PRC": "PRC",
            "train/R": "R",
        }

        for task in getattr(pl_module, "task_names", []):
            mapping[f"train/{task}_loss"] = f"L_{task[:1]}"
            mapping[f"val/{task}_miou"] = "mIoU"
            mapping[f"val/{task}_abs_rel"] = "abs_rel"
            mapping[f"val/{task}_mean_angle"] = "angle"

        new_items = {}
        for k, v in items.items():
            base_k = k.replace("_step", "").replace("_epoch", "")
            if base_k in mapping:
                new_items[mapping[base_k]] = v
            elif base_k.split("/")[-1] in [m.split("/")[-1] for m in mapping if "/" in m]:
                for mk, mv in mapping.items():
                    if mk.endswith(f"/{base_k.split('/')[-1]}"):
                        new_items[mv] = v
                        break
            else:
                clean_k = k.replace("train/", "").replace("val/", "").replace("health/", "")
                new_items[clean_k] = v
        return new_items


class MinimalProgressBar(Callback):
    """
    Log-safe progress reporter using tqdm for in-place terminal updates.

    Updates the tqdm bar at fixed time intervals (not every batch) to keep
    output clean in notebook and hosted runtimes while still showing live
    progress.
    """

    def __init__(self, update_interval_seconds: float = 15.0):
        super().__init__()
        self.update_interval_seconds = max(float(update_interval_seconds), 1.0)
        self._fit_start_time: float | None = None
        self._epoch_start_time: float | None = None
        self._last_emit_time: float = 0.0
        self._bar: tqdm | None = None

    def _should_emit(self) -> bool:
        return (time.time() - self._last_emit_time) >= self.update_interval_seconds

    def _max_epochs(self, trainer: pl.Trainer) -> int | str:
        max_epochs = getattr(trainer, "max_epochs", None)
        if isinstance(max_epochs, int) and max_epochs > 0:
            return max_epochs
        return "?"

    def _num_batches(self, trainer: pl.Trainer) -> int | None:
        num_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(num_batches, int) and num_batches > 0:
            return num_batches
        return None

    def _collect_metrics(self, trainer: pl.Trainer) -> dict[str, str]:
        metrics = trainer.callback_metrics
        result: dict[str, str] = {}

        for key, label in (
            ("train/total_loss", "loss"),
            ("loss", "loss"),
            ("val/total_loss", "vL"),
            ("val/segmentation_miou", "miou"),
            ("val/depth_abs_rel", "abs_rel"),
            ("val/normals_mean_angle", "angle"),
        ):
            value = _safe_float(metrics.get(key))
            if value is not None:
                result[label] = f"{value:.4f}"
                if label == "loss":
                    break

        if "vL" not in result:
            value = _safe_float(metrics.get("val_loss"))
            if value is not None:
                result["vL"] = f"{value:.4f}"

        return result

    def _update_bar(
        self,
        trainer: pl.Trainer,
        batch_idx: int | None = None,
        *,
        prefix: str = "train",
    ) -> None:
        if self._bar is None:
            return

        now = time.time()
        fit_elapsed = None if self._fit_start_time is None else now - self._fit_start_time

        current_epoch = int(trainer.current_epoch) + 1
        max_epochs = self._max_epochs(trainer)
        total_batches = self._num_batches(trainer)

        if total_batches is not None and batch_idx is not None:
            self._bar.n = batch_idx + 1
            self._bar.total = total_batches
        else:
            self._bar.n = int(trainer.global_step)

        self._bar.set_description(f"{prefix} ep={current_epoch}/{max_epochs}")

        postfix_parts: list[str] = []
        if fit_elapsed is not None:
            postfix_parts.append(f"el={_format_seconds(fit_elapsed)}")

        if total_batches is not None and batch_idx is not None and fit_elapsed and trainer.global_step > 0:
            epochs_done = trainer.current_epoch
            completed_steps = epochs_done * total_batches + min(batch_idx + 1, total_batches)
            total_steps = total_batches * max_epochs if isinstance(max_epochs, int) else None
            if total_steps and completed_steps > 0:
                seconds_per_step = fit_elapsed / completed_steps
                postfix_parts.append(f"eta={_format_seconds((total_steps - completed_steps) * seconds_per_step)}")

        metrics = self._collect_metrics(trainer)
        for k, v in metrics.items():
            postfix_parts.append(f"{k}={v}")

        self._bar.set_postfix_str(" ".join(postfix_parts))
        self._bar.refresh()
        self._last_emit_time = time.time()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._fit_start_time = time.time()
        self._last_emit_time = 0.0
        total_batches = self._num_batches(trainer) or 0
        self._bar = tqdm(
            total=total_batches,
            desc="train",
            unit="batch",
            leave=True,
            mininterval=self.update_interval_seconds,
            miniters=1,
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_start_time = time.time()
        total_batches = self._num_batches(trainer) or 0
        if self._bar is not None:
            self._bar.reset(total=total_batches)
        self._update_bar(trainer, batch_idx=0, prefix="train")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._should_emit():
            self._update_bar(trainer, batch_idx=batch_idx, prefix="train")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._update_bar(trainer, prefix="val")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._bar is not None:
            total_batches = self._num_batches(trainer) or 0
            self._bar.n = total_batches
            total_elapsed = None if self._fit_start_time is None else time.time() - self._fit_start_time
            self._bar.set_postfix_str(f"done elapsed={_format_seconds(total_elapsed)}")
            self._bar.close()
            self._bar = None


def build_progress_bar(cfg: DictConfig) -> Callback:
    progress_cfg = cfg.get("train", {}).get("progress", {})
    progress_type = str(progress_cfg.get("type", "sota")).strip().lower()

    if progress_type == "minimal":
        interval = progress_cfg.get("update_interval_seconds", 15.0)
        return MinimalProgressBar(update_interval_seconds=interval)
    if progress_type == "sota":
        refresh_rate = int(progress_cfg.get("refresh_rate", 1))
        return SOTAProgressBar(refresh_rate=refresh_rate)

    raise ValueError(
        f"Unsupported train.progress.type='{progress_type}'. "
        "Expected one of: 'sota', 'minimal'."
    )
