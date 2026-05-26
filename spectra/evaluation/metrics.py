"""
Task-specific evaluation metrics for SPECTRA multi-task benchmarks.

Design principles:
    1. GLOBAL accumulation (not batch-average) — confusion matrix mIoU
       is aggregated over the entire validation set before computing.
       Batch-average mIoU is statistically biased for imbalanced datasets.
    2. Masking is always consistent with loss functions (same invalid-pixel
       exclusion as MaskedL1Loss and DenseCosineLoss).
    3. GPU-first accumulation — all metric state lives on GPU during update()
       to avoid per-batch CUDA→CPU sync stalls. CPU transfer happens only
       once in compute() (a few scalars or a small histogram).
    4. Thread-safe: each metric object is reset at the start of every
       validation epoch via on_validation_epoch_start().

References:
    - Eigen & Fergus "Predicting Depth, Surface Normals..." (ICCV 2015)
    - Liu et al. "MTAN" (CVPR 2019) — standard NYUv2 MTL metrics
    - Long et al. "Fully Convolutional Networks" (CVPR 2015) — mIoU formula
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist


# ===========================================================================
# DISTRIBUTED SYNC HELPERS (DDP)
# ===========================================================================

def sync_tensor_across_gpus(t: torch.Tensor) -> torch.Tensor:
    """All-reduces a tensor across GPUs (sum) if DDP is initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    
    original_device = t.device
    if dist.get_backend() == "nccl" and not t.is_cuda:
        # Move to the current rank's CUDA device for NCCL
        t = t.cuda()
        
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.to(original_device)


def gather_tensor_across_gpus(t: torch.Tensor) -> torch.Tensor:
    """Exact all_gather for variable-length 1D tensors with padding."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
        
    original_device = t.device
    if dist.get_backend() == "nccl" and not t.is_cuda:
        t = t.cuda()
        
    world_size = dist.get_world_size()
    
    # 1. Gather lengths
    local_len = torch.tensor([len(t)], dtype=torch.long, device=t.device)
    lengths = [torch.zeros(1, dtype=torch.long, device=t.device) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    
    max_len = max([l.item() for l in lengths])
    
    # 2. Pad local tensor
    if len(t) < max_len:
        pad_size = max_len - len(t)
        t_padded = torch.cat([t, torch.zeros(pad_size, dtype=t.dtype, device=t.device)])
    else:
        t_padded = t
        
    # 3. All gather padded
    gathered_padded = [torch.zeros(max_len, dtype=t.dtype, device=t.device) for _ in range(world_size)]
    dist.all_gather(gathered_padded, t_padded)
    
    # 4. Unpad and concat
    result = []
    for i in range(world_size):
        valid_len = lengths[i].item()
        result.append(gathered_padded[i][:valid_len])
        
    final_tensor = torch.cat(result)
    return final_tensor.to(original_device)



# ===========================================================================
# SEGMENTATION METRICS
# ===========================================================================

class SegmentationMetrics:
    """
    Global confusion-matrix-based mIoU for semantic segmentation.

    WHY GLOBAL (not batch-average):
        mIoU computed from a global confusion matrix is the correct estimator.
        Batch-average mIoU over-weights batches with more valid pixels and
        under-weights rare categories that may not appear in every batch.
        The difference can be 2–5 mIoU points on NYUv2 — larger than the
        improvement we are trying to demonstrate.

    Usage:
        metrics = SegmentationMetrics(num_classes=13, ignore_index=255)
        for batch in val_loader:
            metrics.update(pred_logits, target)
        results = metrics.compute()  # {"miou": float, "acc": float, ...}
        metrics.reset()
    """

    def __init__(self, num_classes: int = 13, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._device = None
        self._conf_matrix: Optional[torch.Tensor] = None

    def _lazy_init(self, device: torch.device) -> None:
        """Allocate confusion matrix on the same device as input (GPU)."""
        self._device = device
        self._conf_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long, device=device
        )

    def reset(self) -> None:
        """Call at the start of every validation epoch."""
        if self._conf_matrix is not None:
            self._conf_matrix.zero_()

    @torch.no_grad()
    def update(self, pred_logits: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate predictions into global confusion matrix on GPU.

        Args:
            pred_logits: [B, C, H, W] raw logits (before softmax).
            target:      [B, H, W] ground-truth class indices (int64).
                         Pixels with target == ignore_index are excluded.
        """
        # Lazy-init on input device
        if self._conf_matrix is None or self._device != pred_logits.device:
            self._lazy_init(pred_logits.device)

        pred = pred_logits.argmax(dim=1).detach()  # [B, H, W] — stays on GPU
        target = target.detach()

        # Flatten and filter ignored pixels — all on GPU
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        valid_mask  = target_flat != self.ignore_index

        pred_valid   = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        # Build confusion matrix using scatter-add (torch.bincount works on GPU)
        # conf[i, j] = number of pixels with true class i predicted as class j
        indices = target_valid * self.num_classes + pred_valid
        conf_flat = torch.bincount(indices, minlength=self.num_classes ** 2)
        self._conf_matrix += conf_flat.view(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        """
        Compute mIoU, mean accuracy, per-class IoU from the accumulated matrix.

        Returns:
            dict with keys: miou, pixel_acc, mean_class_acc, per_class_iou (List)
        """
        if self._conf_matrix is None:
            conf = torch.zeros(self.num_classes, self.num_classes)
        else:
            # Single CPU transfer — C×C matrix (13×13 = 169 scalars)
            conf = self._conf_matrix.float().cpu()

        # DDP: all-reduce the confusion matrix globally for exact metrics.
        conf = sync_tensor_across_gpus(conf)

        # True positives: diagonal
        tp = conf.diag()

        # Per-class IoU: TP / (TP + FP + FN)
        # FP for class i = sum of column i minus TP
        # FN for class i = sum of row i minus TP
        # IoU_i = TP_i / (row_sum_i + col_sum_i - TP_i)
        row_sum = conf.sum(dim=1)  # Ground truth counts per class
        col_sum = conf.sum(dim=0)  # Prediction counts per class
        denom   = row_sum + col_sum - tp

        # Only compute IoU for classes that appear in the ground truth
        valid_classes = row_sum > 0
        iou_per_class = torch.zeros(self.num_classes)
        iou_per_class[valid_classes] = tp[valid_classes] / denom[valid_classes].clamp(min=1.0)

        miou = iou_per_class[valid_classes].mean().item()

        # Pixel accuracy
        total_pixels = conf.sum().item()
        correct_pixels = tp.sum().item()
        pixel_acc = correct_pixels / max(total_pixels, 1.0)

        # Mean class accuracy
        class_acc = torch.zeros(self.num_classes)
        class_acc[valid_classes] = tp[valid_classes] / row_sum[valid_classes].clamp(min=1.0)
        mean_class_acc = class_acc[valid_classes].mean().item()

        return {
            "miou":           miou,
            "pixel_acc":      pixel_acc,
            "mean_class_acc": mean_class_acc,
            "per_class_iou":  iou_per_class.tolist(),
            "n_valid_classes": valid_classes.sum().item(),
        }


# ===========================================================================
# DEPTH ESTIMATION METRICS
# ===========================================================================

class DepthMetrics:
    """
    Standard depth estimation metrics (Eigen & Fergus 2015, MTAN Liu 2019).

    Metrics computed:
        - abs_rel:  mean |pred - gt| / gt                    (lower = better)
        - sq_rel:   mean |pred - gt|^2 / gt                  (lower = better)
        - rmse:     sqrt(mean (pred - gt)^2)                  (lower = better)
        - log_rmse: sqrt(mean (log pred - log gt)^2)          (lower = better)
        - delta_1:  fraction where max(pred/gt, gt/pred) < 1.25   (higher = better)
        - delta_2:  fraction where max(pred/gt, gt/pred) < 1.25^2 (higher = better)
        - delta_3:  fraction where max(pred/gt, gt/pred) < 1.25^3 (higher = better)

    All computed on valid pixels only (depth > 0).

    Performance design:
        Computation stays on GPU (no per-batch CPU transfer of raw values).
        Online accumulation of 8 running stats — O(1) memory vs O(N).
        Only the final 8-scalar vector is transferred to CPU at compute().
    """

    def __init__(self, max_depth: float = 10.0):
        """
        Args:
            max_depth: Clip predictions to [0, max_depth] before metrics.
                       NYUv2 standard: 10.0 meters.
        """
        self.max_depth = max_depth
        self._device = None
        self._stats: Optional[torch.Tensor] = None

    def _lazy_init(self, device: torch.device) -> None:
        """Allocate accumulators on the same device as input (GPU)."""
        self._device = device
        self._stats = torch.zeros(8, dtype=torch.float64, device=device)

    def reset(self) -> None:
        if self._stats is not None:
            self._stats.zero_()

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Accumulate depth stats entirely on GPU (no CPU sync in hot path).

        Args:
            pred:   [B, 1, H, W] predicted depth in meters (may be negative pre-clip).
            target: [B, 1, H, W] ground-truth depth in meters (0 = invalid).
            mask:   [B, 1, H, W] float mask (1=valid, 0=invalid). If None,
                    derived from target > 0.
        """
        pred   = pred.detach().float()
        target = target.detach().float()

        # Lazy-init accumulators on input device
        if self._stats is None or self._device != pred.device:
            self._lazy_init(pred.device)

        if mask is None:
            mask = (target > 0.0).float()
        else:
            mask = mask.detach().float()

        # Clip predictions to valid depth range
        pred = pred.clamp(min=1e-3, max=self.max_depth)

        # Extract valid pixels
        pred_flat   = pred[mask > 0.5]
        target_flat = target[mask > 0.5]

        if pred_flat.numel() == 0:
            return

        # Clamp for log safety
        pred_clamped = pred_flat.clamp(min=1e-3)
        target_clamped = target_flat.clamp(min=1e-3)

        # Compute per-batch stats — ALL on GPU, no sync stall
        batch_stats = torch.stack([
            (torch.abs(pred_flat - target_flat) / target_clamped).sum(),
            ((pred_flat - target_flat) ** 2 / target_clamped).sum(),
            ((pred_flat - target_flat) ** 2).sum(),
            ((torch.log(pred_clamped) - torch.log(target_clamped)) ** 2).sum(),
            (torch.max(pred_clamped / target_clamped, target_clamped / pred_clamped) < 1.25).float().sum(),
            (torch.max(pred_clamped / target_clamped, target_clamped / pred_clamped) < 1.25 ** 2).float().sum(),
            (torch.max(pred_clamped / target_clamped, target_clamped / pred_clamped) < 1.25 ** 3).float().sum(),
            pred_flat.new_tensor(float(pred_flat.numel()), dtype=torch.float64),
        ]).double()

        self._stats += batch_stats

    def compute(self) -> Dict[str, float]:
        """Compute all depth metrics from accumulated stats."""
        if self._stats is None:
            stats = torch.zeros(8, dtype=torch.float32)
        else:
            # Single CPU transfer — 8 scalars
            stats = self._stats.float().cpu()

        # We MUST sync even if local stats are zero to prevent DDP deadlock!
        stats_synced = sync_tensor_across_gpus(stats)

        n_v = stats_synced[7].item()
        if n_v < 1.0:
             return {k: float("nan") for k in ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3"]}

        return {
            "abs_rel":  stats_synced[0].item() / n_v,
            "sq_rel":   stats_synced[1].item() / n_v,
            "rmse":     math.sqrt(stats_synced[2].item() / n_v),
            "log_rmse": math.sqrt(stats_synced[3].item() / n_v),
            "delta_1":  stats_synced[4].item() / n_v,
            "delta_2":  stats_synced[5].item() / n_v,
            "delta_3":  stats_synced[6].item() / n_v,
            "n_valid":  int(n_v),
        }


# ===========================================================================
# SURFACE NORMAL METRICS
# ===========================================================================

class NormalMetrics:
    """
    Angular error metrics for surface normal estimation.

    Metrics computed (MTAN Liu 2019 standard):
        - mean_angle_deg:   mean angular error in degrees  (lower = better)
        - median_angle_deg: median angular error in degrees (lower = better)
        - within_11_25:     fraction with angle error < 11.25° (higher = better)
        - within_22_5:      fraction with angle error < 22.5°  (higher = better)
        - within_30:        fraction with angle error < 30°     (higher = better)

    Invalid normals (|gt| = 0) are automatically excluded.

    Performance design:
        All computation stays on GPU (no per-batch CPU transfer).
        Median is approximated via 360-bin histogram over [0°, 180°]
        (0.5° resolution) using torch.histc — O(1) memory vs O(N).
        Mean and threshold fractions use exact running sums.
    """

    # Histogram bins for median approximation: 360 bins × 0.5° over [0, 180]
    _HIST_BINS = 360
    _HIST_MAX = 180.0
    _RAD2DEG = 180.0 / math.pi

    def __init__(self):
        self._device = None  # lazy init on first update
        self._angle_sum: Optional[torch.Tensor] = None
        self._n_valid: Optional[torch.Tensor] = None
        self._within_11_25: Optional[torch.Tensor] = None
        self._within_22_5: Optional[torch.Tensor] = None
        self._within_30: Optional[torch.Tensor] = None
        self._histogram: Optional[torch.Tensor] = None

    def _lazy_init(self, device: torch.device) -> None:
        """Allocate accumulators on the same device as input (GPU)."""
        self._device = device
        self._angle_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        self._n_valid = torch.tensor(0.0, dtype=torch.float64, device=device)
        self._within_11_25 = torch.tensor(0.0, dtype=torch.float64, device=device)
        self._within_22_5 = torch.tensor(0.0, dtype=torch.float64, device=device)
        self._within_30 = torch.tensor(0.0, dtype=torch.float64, device=device)
        self._histogram = torch.zeros(self._HIST_BINS, dtype=torch.float64, device=device)

    def reset(self) -> None:
        if self._angle_sum is not None:
            self._angle_sum.zero_()
            self._n_valid.zero_()
            self._within_11_25.zero_()
            self._within_22_5.zero_()
            self._within_30.zero_()
            self._histogram.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate angular errors entirely on GPU (no CPU sync in hot path).

        Args:
            pred:   [B, 3, H, W] predicted normal vectors (need not be unit length).
            target: [B, 3, H, W] ground-truth normal vectors.
                    Zero-vectors indicate invalid regions (excluded).
        """
        pred   = pred.detach().float()
        target = target.detach().float()

        # Lazy-init accumulators on input device
        if self._angle_sum is None or self._device != pred.device:
            self._lazy_init(pred.device)

        # Valid pixel mask: ground-truth normals with nonzero magnitude
        target_norm = torch.norm(target, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
        valid_mask  = (target_norm > 1e-6).squeeze(1)               # [B, H, W]

        # Normalize both to unit vectors
        pred_unit   = F.normalize(pred,   p=2, dim=1)  # [B, 3, H, W]
        target_unit = F.normalize(target, p=2, dim=1)  # [B, 3, H, W]

        # Cosine similarity per pixel: dot product along channel dim
        cos_sim = (pred_unit * target_unit).sum(dim=1)  # [B, H, W]
        # Clamp for numerical safety before acos
        cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Angular error in degrees — ALL on GPU
        angle_deg = torch.acos(cos_sim) * self._RAD2DEG  # [B, H, W]

        # Extract valid pixels only
        valid_angles = angle_deg[valid_mask]  # [N_valid]
        n = valid_angles.numel()
        if n == 0:
            return

        # Accumulate running stats — GPU→GPU, no sync stall
        self._angle_sum += valid_angles.sum().double()
        self._n_valid += valid_angles.new_tensor(n, dtype=torch.float64)

        # Threshold fractions (exact counts on GPU)
        self._within_11_25 += (valid_angles < 11.25).sum().double()
        self._within_22_5  += (valid_angles < 22.5 ).sum().double()
        self._within_30    += (valid_angles < 30.0 ).sum().double()

        # Histogram for median — torch.histc runs natively on GPU
        hist_batch = torch.histc(
            valid_angles.double(),
            bins=self._HIST_BINS,
            min=0.0,
            max=self._HIST_MAX,
        )
        self._histogram += hist_batch

    def compute(self) -> Dict[str, float]:
        """Compute all normal metrics from accumulated stats + histogram."""
        if self._angle_sum is None or self._n_valid.item() < 1.0:
            return {k: float("nan") for k in
                    ["mean_angle_deg", "median_angle_deg",
                     "within_11_25", "within_22_5", "within_30"]}

        # Single CPU transfer — 5 scalars + 360-bin histogram
        stats = torch.stack([
            self._angle_sum,
            self._n_valid,
            self._within_11_25,
            self._within_22_5,
            self._within_30,
        ]).float().cpu()
        stats = sync_tensor_across_gpus(stats)
        hist_synced = sync_tensor_across_gpus(self._histogram.float().cpu())

        n_v = stats[1].item()
        if n_v < 1.0:
            return {k: float("nan") for k in
                    ["mean_angle_deg", "median_angle_deg",
                     "within_11_25", "within_22_5", "within_30"]}

        # Mean (exact)
        mean_angle = stats[0].item() / n_v

        # Median via histogram CDF
        cumsum = hist_synced.cumsum(0)
        half = n_v / 2.0
        median_bin = torch.searchsorted(cumsum, torch.tensor(half)).item()
        median_bin = min(median_bin, self._HIST_BINS - 1)
        bin_width = self._HIST_MAX / self._HIST_BINS
        median_angle = (median_bin + 0.5) * bin_width  # center of bin

        return {
            "mean_angle_deg":   mean_angle,
            "median_angle_deg": median_angle,
            "within_11_25":     stats[2].item() / n_v,
            "within_22_5":      stats[3].item() / n_v,
            "within_30":        stats[4].item() / n_v,
            "n_valid":          int(n_v),
        }


# ===========================================================================
# DELTA-M% UTILITY (Multi-Task Performance Improvement Metric)
# ===========================================================================

def compute_delta_m(
    results: Dict[str, float],
    baseline: Dict[str, float],
    metric_configs: Dict[str, bool],
) -> float:
    """
    Compute Δm% relative improvement metric (Liu et al. MTAN 2019).

    Δm = (1/T) Σ_i ((-1)^l_i * (result_i - baseline_i) / baseline_i) * 100

    where l_i = 1 if lower is better for metric i, 0 otherwise.

    Args:
        results:        Current method metrics {"metric_name": value}.
        baseline:       Single-task learning / reference baseline metrics.
        metric_configs: {"metric_name": lower_is_better} bool per metric.

    Returns:
        Δm% (positive = better than baseline).

    Example:
        delta_m = compute_delta_m(
            results={"miou": 0.45, "abs_rel": 0.18, "mean_angle_deg": 25.0},
            baseline={"miou": 0.40, "abs_rel": 0.20, "mean_angle_deg": 27.0},
            metric_configs={"miou": False, "abs_rel": True, "mean_angle_deg": True},
        )
        # Returns positive value if current method beats baseline.
    """
    improvements = []
    for metric, lower_is_better in metric_configs.items():
        if metric not in results or metric not in baseline:
            continue
        b = baseline[metric]
        r = results[metric]
        if abs(b) < 1e-10:
            continue
        relative = (b - r) / b if lower_is_better else (r - b) / b
        improvements.append(relative)

    if not improvements:
        return float("nan")

    return (sum(improvements) / len(improvements)) * 100.0


# Standard NYUv2 Δm% metric config (matches MTAN paper Table 1)
NYUv2_DELTA_M_METRICS = {
    "miou":           False,  # Higher is better
    "pixel_acc":      False,  # Higher is better
    "abs_rel":        True,   # Lower is better
    "delta_1":        False,  # Higher is better
    "mean_angle_deg": True,   # Lower is better
    "within_11_25":   False,  # Higher is better
}


# ===========================================================================
# TABULAR REGRESSION METRICS
# ===========================================================================

class RegressionMetrics:
    """
    Exact global regression metrics accumulated over a full validation epoch.

    Metrics:
        - mae:  mean absolute error
        - rmse: root mean squared error
        - r2:   coefficient of determination

    Accumulation is exact over the entire validation set rather than averaging
    per-batch metrics, which would bias results when the last batch is smaller.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._abs_error_sum = torch.tensor(0.0, dtype=torch.float64)
        self._sq_error_sum = torch.tensor(0.0, dtype=torch.float64)
        self._target_sum = torch.tensor(0.0, dtype=torch.float64)
        self._target_sq_sum = torch.tensor(0.0, dtype=torch.float64)
        self._count = torch.tensor(0.0, dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().double().reshape(-1).cpu()
        target = target.detach().double().reshape(-1).cpu()
        if pred.numel() == 0:
            return

        diff = pred - target
        self._abs_error_sum += diff.abs().sum()
        self._sq_error_sum += diff.square().sum()
        self._target_sum += target.sum()
        self._target_sq_sum += target.square().sum()
        self._count += torch.tensor(float(target.numel()), dtype=torch.float64)

    def compute(self) -> Dict[str, float]:
        stats = torch.stack(
            [
                self._abs_error_sum,
                self._sq_error_sum,
                self._target_sum,
                self._target_sq_sum,
                self._count,
            ]
        ).float()
        stats = sync_tensor_across_gpus(stats)

        count = float(stats[4].item())
        if count <= 0.0:
            return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}

        abs_error_sum = float(stats[0].item())
        sq_error_sum = float(stats[1].item())
        target_sum = float(stats[2].item())
        target_sq_sum = float(stats[3].item())

        mae = abs_error_sum / count
        rmse = math.sqrt(sq_error_sum / count)

        target_mean = target_sum / count
        ss_tot = target_sq_sum - (count * target_mean * target_mean)
        if ss_tot <= 1e-12:
            r2 = 1.0 if sq_error_sum <= 1e-12 else 0.0
        else:
            r2 = 1.0 - (sq_error_sum / ss_tot)

        return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(count)}


class BinaryClassificationMetrics:
    """
    Exact global binary classification metrics from logits.

    Metrics:
        - accuracy
        - precision
        - recall
        - f1
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._tp = torch.tensor(0.0, dtype=torch.float64)
        self._fp = torch.tensor(0.0, dtype=torch.float64)
        self._fn = torch.tensor(0.0, dtype=torch.float64)
        self._tn = torch.tensor(0.0, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probs = torch.sigmoid(logits.detach()).reshape(-1).cpu()
        pred = probs >= self.threshold
        truth = target.detach().reshape(-1).cpu() >= 0.5

        self._tp += (pred & truth).sum().double()
        self._fp += (pred & ~truth).sum().double()
        self._fn += (~pred & truth).sum().double()
        self._tn += (~pred & ~truth).sum().double()

    def compute(self) -> Dict[str, float]:
        stats = torch.stack([self._tp, self._fp, self._fn, self._tn]).float()
        stats = sync_tensor_across_gpus(stats)
        tp, fp, fn, tn = [float(x.item()) for x in stats]

        total = tp + fp + fn + tn
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        accuracy = (tp + tn) / max(total, 1.0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": int(total),
        }


class MultiLabelClassificationMetrics:
    """
    Exact global multilabel metrics from stacked logits and binary targets.

    Metrics:
        - micro_f1
        - macro_f1
        - hamming_acc
        - subset_acc
    """

    def __init__(self, num_labels: int, threshold: float = 0.5):
        self.num_labels = num_labels
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._tp = torch.zeros(self.num_labels, dtype=torch.float64)
        self._fp = torch.zeros(self.num_labels, dtype=torch.float64)
        self._fn = torch.zeros(self.num_labels, dtype=torch.float64)
        self._tn = torch.zeros(self.num_labels, dtype=torch.float64)
        self._subset_correct = torch.tensor(0.0, dtype=torch.float64)
        self._sample_count = torch.tensor(0.0, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probs = torch.sigmoid(logits.detach()).cpu()
        pred = probs >= self.threshold
        truth = target.detach().cpu() >= 0.5

        self._tp += (pred & truth).sum(dim=0).double()
        self._fp += (pred & ~truth).sum(dim=0).double()
        self._fn += (~pred & truth).sum(dim=0).double()
        self._tn += (~pred & ~truth).sum(dim=0).double()
        self._subset_correct += (pred == truth).all(dim=1).sum().double()
        self._sample_count += torch.tensor(float(truth.shape[0]), dtype=torch.float64)

    def compute(self) -> Dict[str, float]:
        tp = sync_tensor_across_gpus(self._tp.float()).double()
        fp = sync_tensor_across_gpus(self._fp.float()).double()
        fn = sync_tensor_across_gpus(self._fn.float()).double()
        tn = sync_tensor_across_gpus(self._tn.float()).double()
        subset_correct = float(sync_tensor_across_gpus(self._subset_correct.float()).item())
        sample_count = float(sync_tensor_across_gpus(self._sample_count.float()).item())

        per_label_precision = tp / torch.clamp(tp + fp, min=1.0)
        per_label_recall = tp / torch.clamp(tp + fn, min=1.0)
        per_label_f1 = (2.0 * per_label_precision * per_label_recall) / torch.clamp(
            per_label_precision + per_label_recall,
            min=1e-12,
        )

        tp_sum = float(tp.sum().item())
        fp_sum = float(fp.sum().item())
        fn_sum = float(fn.sum().item())
        total_labels = float((tp + fp + fn + tn).sum().item())

        micro_precision = tp_sum / max(tp_sum + fp_sum, 1.0)
        micro_recall = tp_sum / max(tp_sum + fn_sum, 1.0)
        micro_f1 = (2.0 * micro_precision * micro_recall) / max(micro_precision + micro_recall, 1e-12)
        hamming_acc = float((tp + tn).sum().item()) / max(total_labels, 1.0)
        subset_acc = subset_correct / max(sample_count, 1.0)

        return {
            "micro_f1": micro_f1,
            "macro_f1": float(per_label_f1.mean().item()),
            "hamming_acc": hamming_acc,
            "subset_acc": subset_acc,
            "n_samples": int(sample_count),
        }


# ===========================================================================
# STANDALONE VERIFICATION
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Metrics — Verification")
    print("=" * 60)
    torch.manual_seed(42)

    B, H, W, C = 2, 32, 32, 13

    # ── SegmentationMetrics ──
    print("\n[SegmentationMetrics]")
    seg = SegmentationMetrics(num_classes=C)

    # Perfect prediction: pred == target → mIoU = 1.0
    target = torch.randint(0, C, (B, H, W))
    logits = torch.zeros(B, C, H, W)
    logits.scatter_(1, target.unsqueeze(1), 100.0)  # One-hot logits

    seg.update(logits, target)
    r = seg.compute()
    assert abs(r["miou"] - 1.0) < 1e-4, f"Perfect prediction mIoU should be 1.0, got {r['miou']}"
    print(f"  ✓ Perfect prediction mIoU = {r['miou']:.4f}")

    # Global vs batch-average test (imbalanced classes)
    seg.reset()
    target_a = torch.zeros(1, H, W, dtype=torch.long)     # All class 0
    target_b = torch.ones(1, H, W, dtype=torch.long)      # All class 1
    logits_a = torch.zeros(1, C, H, W); logits_a[0, 0] = 10  # Predicts class 0
    logits_b = torch.zeros(1, C, H, W); logits_b[0, 1] = 10  # Predicts class 1
    seg.update(logits_a, target_a)
    seg.update(logits_b, target_b)
    r2 = seg.compute()
    print(f"  ✓ Two-class mIoU (perfect): {r2['miou']:.4f} (should be 1.0)")
    assert abs(r2["miou"] - 1.0) < 1e-4

    # ── DepthMetrics ──
    print("\n[DepthMetrics]")
    depth_m = DepthMetrics(max_depth=10.0)

    # Perfect prediction
    target_d = torch.rand(B, 1, H, W) * 5.0 + 0.5
    pred_d   = target_d.clone()
    depth_m.update(pred_d, target_d)
    r_d = depth_m.compute()
    assert r_d["abs_rel"] < 1e-5, f"Perfect depth: abs_rel={r_d['abs_rel']}"
    assert r_d["delta_1"] > 0.999, f"Perfect depth: delta_1={r_d['delta_1']}"
    print(f"  ✓ Perfect depth: abs_rel={r_d['abs_rel']:.6f}, δ<1.25={r_d['delta_1']:.4f}")

    # Masked invalid pixels
    depth_m.reset()
    target_d_masked = torch.rand(B, 1, H, W) * 5.0
    target_d_masked[:, :, :5, :5] = 0.0  # Invalid region
    depth_m.update(torch.zeros_like(target_d_masked), target_d_masked)
    r_invalid = depth_m.compute()
    n_total = B * H * W
    n_invalid = B * 5 * 5
    assert r_invalid["n_valid"] == n_total - n_invalid, "Invalid pixel masking failed"
    print(f"  ✓ Invalid pixel masking: {r_invalid['n_valid']} valid / {n_total} total")

    # ── NormalMetrics ──
    print("\n[NormalMetrics]")
    norm_m = NormalMetrics()

    # Perfect prediction (same direction)
    target_n = F.normalize(torch.randn(B, 3, H, W), p=2, dim=1)
    norm_m.update(target_n, target_n)
    r_n = norm_m.compute()
    assert r_n["mean_angle_deg"] < 0.1, f"Perfect normals angle: {r_n['mean_angle_deg']}"
    print(f"  ✓ Perfect normals: mean_angle={r_n['mean_angle_deg']:.4f}°")

    # Opposite direction (should be ~180°)
    norm_m.reset()
    norm_m.update(-target_n, target_n)
    r_n_opp = norm_m.compute()
    assert r_n_opp["mean_angle_deg"] > 175.0, f"Opposite normals angle: {r_n_opp['mean_angle_deg']}"
    print(f"  ✓ Opposite normals: mean_angle={r_n_opp['mean_angle_deg']:.2f}° (≈180°)")

    # ── Δm% ──
    print("\n[compute_delta_m]")
    dm = compute_delta_m(
        results  = {"miou": 0.45, "abs_rel": 0.18, "mean_angle_deg": 25.0},
        baseline = {"miou": 0.40, "abs_rel": 0.20, "mean_angle_deg": 27.0},
        metric_configs = {"miou": False, "abs_rel": True, "mean_angle_deg": True},
    )
    print(f"  Δm% = {dm:.2f}% (positive = better than baseline)")
    assert dm > 0, "Should be positive improvement"
    print(f"  ✓ Δm% computed correctly")

    print("\n" + "=" * 60)
    print("All checks PASSED.")
    print("=" * 60)
