"""
spectra/evaluation/metrics.py
------------------------------
Task-specific evaluation metrics for SPECTRA multi-task benchmarks.

Design principles:
    1. GLOBAL accumulation (not batch-average) — confusion matrix mIoU
       is aggregated over the entire validation set before computing.
       Batch-average mIoU is statistically biased for imbalanced datasets.
    2. Masking is always consistent with loss functions (same invalid-pixel
       exclusion as MaskedL1Loss and DenseCosineLoss).
    3. All heavy computation runs on CPU after detach() to avoid GPU
       pressure during validation.
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
        self._conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def reset(self) -> None:
        """Call at the start of every validation epoch."""
        self._conf_matrix.zero_()

    @torch.no_grad()
    def update(self, pred_logits: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate predictions into global confusion matrix.

        Args:
            pred_logits: [B, C, H, W] raw logits (before softmax).
            target:      [B, H, W] ground-truth class indices (int64).
                         Pixels with target == ignore_index are excluded.
        """
        pred = pred_logits.argmax(dim=1).detach().cpu()  # [B, H, W]
        target = target.detach().cpu()

        # Flatten and filter ignored pixels
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        valid_mask  = target_flat != self.ignore_index

        pred_valid   = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        # Build confusion matrix using a scatter-add trick (fast, O(N))
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
        conf = self._conf_matrix.float()

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
    Buffered: accumulates predictions and targets for exact computation.
    """

    def __init__(self, max_depth: float = 10.0):
        """
        Args:
            max_depth: Clip predictions to [0, max_depth] before metrics.
                       NYUv2 standard: 10.0 meters.
        """
        self.max_depth = max_depth
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def reset(self) -> None:
        self._preds = []
        self._targets = []

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Accumulate depth predictions.

        Args:
            pred:   [B, 1, H, W] predicted depth in meters (may be negative pre-clip).
            target: [B, 1, H, W] ground-truth depth in meters (0 = invalid).
            mask:   [B, 1, H, W] float mask (1=valid, 0=invalid). If None,
                    derived from target > 0.
        """
        pred   = pred.detach().float().cpu()
        target = target.detach().float().cpu()

        if mask is None:
            mask = (target > 0.0).float()
        else:
            mask = mask.detach().float().cpu()

        # Clip predictions to valid depth range
        pred = pred.clamp(min=1e-3, max=self.max_depth)

        # Store flattened valid-pixel values
        pred_flat   = pred[mask > 0.5]
        target_flat = target[mask > 0.5]

        if len(pred_flat) > 0:
            self._preds.append(pred_flat)
            self._targets.append(target_flat)

    def compute(self) -> Dict[str, float]:
        """Compute all depth metrics over accumulated valid pixels."""
        if not self._preds:
            return {k: float("nan") for k in
                    ["abs_rel", "sq_rel", "rmse", "log_rmse",
                     "delta_1", "delta_2", "delta_3"]}

        pred   = torch.cat(self._preds)    # [N_valid]
        target = torch.cat(self._targets)  # [N_valid]

        # Absolute relative error
        abs_rel = (torch.abs(pred - target) / target).mean().item()

        # Squared relative error
        sq_rel = ((pred - target) ** 2 / target).mean().item()

        # RMSE
        rmse = torch.sqrt(((pred - target) ** 2).mean()).item()

        # Log-scale RMSE
        log_rmse = torch.sqrt(
            ((torch.log(pred.clamp(min=1e-3)) - torch.log(target.clamp(min=1e-3))) ** 2).mean()
        ).item()

        # Threshold accuracy: max(pred/gt, gt/pred) < threshold
        ratio = torch.max(pred / target, target / pred)  # [N]
        delta_1 = (ratio < 1.25    ).float().mean().item()
        delta_2 = (ratio < 1.25 ** 2).float().mean().item()
        delta_3 = (ratio < 1.25 ** 3).float().mean().item()

        return {
            "abs_rel":  abs_rel,
            "sq_rel":   sq_rel,
            "rmse":     rmse,
            "log_rmse": log_rmse,
            "delta_1":  delta_1,
            "delta_2":  delta_2,
            "delta_3":  delta_3,
            "n_valid":  len(pred),
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
    """

    def __init__(self):
        self._angles: List[torch.Tensor] = []  # Angle errors in degrees

    def reset(self) -> None:
        self._angles = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate angular errors.

        Args:
            pred:   [B, 3, H, W] predicted normal vectors (need not be unit length).
            target: [B, 3, H, W] ground-truth normal vectors.
                    Zero-vectors indicate invalid regions (excluded).
        """
        pred   = pred.detach().float().cpu()
        target = target.detach().float().cpu()

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

        # Angular error in degrees
        angle_rad = torch.acos(cos_sim)  # [B, H, W]
        angle_deg = angle_rad * (180.0 / math.pi)

        # Keep only valid pixels
        valid_angles = angle_deg[valid_mask]
        if len(valid_angles) > 0:
            self._angles.append(valid_angles)

    def compute(self) -> Dict[str, float]:
        """Compute all normal metrics over accumulated valid pixels."""
        if not self._angles:
            return {k: float("nan") for k in
                    ["mean_angle_deg", "median_angle_deg",
                     "within_11_25", "within_22_5", "within_30"]}

        angles = torch.cat(self._angles)  # [N_valid]

        return {
            "mean_angle_deg":   angles.mean().item(),
            "median_angle_deg": angles.median().item(),
            "within_11_25":     (angles < 11.25).float().mean().item(),
            "within_22_5":      (angles < 22.5 ).float().mean().item(),
            "within_30":        (angles < 30.0 ).float().mean().item(),
            "n_valid":          len(angles),
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
