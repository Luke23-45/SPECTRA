"""Dense pixel-wise loss functions for NYUv2 multi-task learning.

These losses handle the pixel-level subtleties that standard PyTorch losses
don't support out of the box:
    1. MaskedL1Loss: Depth L1 only on valid pixels (Kinect failures → depth=0)
    2. DenseCosineLoss: Angular error for surface normals with masking

Design note:
    These losses return a single scalar per task. The masking is internal
    to the loss; the weighter sees only the final scalar.

References:
    - Liu et al. "End-to-End Multi-Task Learning with Attention" (CVPR 2019)
    - Eigen & Fergus "Predicting Depth, Surface Normals..." (ICCV 2015)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional


class MaskedL1Loss(nn.Module):
    """
    L1 loss computed only on valid (non-zero) depth pixels.

    The NYUv2 Kinect sensor produces invalid readings (depth=0) where the
    infrared pattern fails. Including these in the loss would teach the model
    to predict 0 depth — corrupting the depth manifold.

    Formula: L = (1/N_valid) * Σ |pred_i - target_i| for all i where target_i > 0

    Attributes:
        requires_mask: Flag for trainer to know this loss needs meta dict.
    """

    requires_mask = True

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   [B, 1, H, W] predicted depth.
            target: [B, 1, H, W] ground-truth depth.
            meta:   Dict with 'depth_mask' [B, 1, H, W] (1=valid, 0=invalid).

        Returns:
            Scalar loss value.
        """
        # Get validity mask
        if meta is not None and "depth_mask" in meta:
            mask = meta["depth_mask"].to(pred.device)
        else:
            # Fallback: derive mask from target (depth > 0 is valid)
            mask = (target > 0.0).float()

        # Ensure shapes match
        if mask.shape != pred.shape:
            # Handle case where mask is [B, H, W] vs pred [B, 1, H, W]
            if mask.dim() == 3 and pred.dim() == 4:
                mask = mask.unsqueeze(1)

        # Compute masked L1
        diff = torch.abs(pred - target)
        masked_diff = diff * mask

        # Normalize by number of valid pixels (prevent division by zero)
        n_valid = mask.sum().clamp(min=1.0)
        loss = masked_diff.sum() / n_valid

        return loss


class DenseCosineLoss(nn.Module):
    """
    Cosine angular error loss for surface normal prediction.

    Measures the angular deviation between predicted and ground-truth
    surface normals at each pixel. Lower is better.

    Formula: L = (1/N_valid) * Σ (1 - cos(pred_i, target_i))

    This is equivalent to the mean angle error (in cosine space) and is
    the standard loss for surface normal estimation (Eigen & Fergus 2015).

    Invalid normals (all zeros) are automatically excluded.

    Attributes:
        requires_mask: Flag for trainer to know this loss may need meta dict.
    """

    requires_mask = False  # We derive mask from target directly

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   [B, 3, H, W] predicted normals.
            target: [B, 3, H, W] ground-truth normals.
            meta:   Optional, unused.

        Returns:
            Scalar loss value (mean 1 - cosine_similarity).
        """
        # Valid pixel mask: normals with non-zero magnitude
        # (zero normals indicate invalid/unmeasured regions)
        target_norm = torch.norm(target, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
        valid_mask = (target_norm > 1e-6).float()

        # Normalize predictions and targets to unit length
        pred_normalized = F.normalize(pred, p=2, dim=1)    # [B, 3, H, W]
        target_normalized = F.normalize(target, p=2, dim=1)  # [B, 3, H, W]

        # Cosine similarity per pixel: dot product along channel dim
        cos_sim = (pred_normalized * target_normalized).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Clamp to [-1, 1] for numerical stability
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Loss: 1 - cos_sim (0 = perfect alignment, 2 = opposite direction)
        pixel_loss = (1.0 - cos_sim) * valid_mask

        # Mean over valid pixels
        n_valid = valid_mask.sum().clamp(min=1.0)
        loss = pixel_loss.sum() / n_valid

        return loss


# =============================================================================
# STANDALONE VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dense Losses — Verification")
    print("=" * 60)

    B, H, W = 2, 32, 32

    # --- MaskedL1Loss ---
    print("\n[MaskedL1Loss]")
    masked_l1 = MaskedL1Loss()

    pred_depth = torch.randn(B, 1, H, W)
    target_depth = torch.rand(B, 1, H, W) * 5.0
    # Inject some invalid pixels
    target_depth[:, :, :5, :5] = 0.0
    depth_mask = (target_depth > 0.0).float()

    loss_d = masked_l1(pred_depth, target_depth, meta={"depth_mask": depth_mask})
    print(f"  Loss (with mask): {loss_d.item():.4f}")
    assert loss_d.isfinite(), "MaskedL1Loss produced non-finite value"
    print("  ✓ Finite loss")

    # Verify zero-masked pixels don't contribute
    all_zero = torch.zeros(B, 1, H, W)
    loss_zero = masked_l1(pred_depth, all_zero, meta={"depth_mask": torch.zeros(B, 1, H, W)})
    assert loss_zero.item() == 0.0, f"Expected 0.0 loss for all-masked, got {loss_zero.item()}"
    print("  ✓ All-masked → loss=0")

    # --- DenseCosineLoss ---
    print("\n[DenseCosineLoss]")
    cosine_loss = DenseCosineLoss()

    pred_normal = torch.randn(B, 3, H, W)
    target_normal = torch.randn(B, 3, H, W)

    loss_n = cosine_loss(pred_normal, target_normal)
    print(f"  Loss (random): {loss_n.item():.4f}")
    assert loss_n.isfinite(), "DenseCosineLoss produced non-finite value"
    print("  ✓ Finite loss")

    # Perfect prediction → loss ≈ 0
    loss_perfect = cosine_loss(target_normal, target_normal)
    assert loss_perfect.item() < 0.01, f"Perfect prediction should have ~0 loss, got {loss_perfect.item()}"
    print(f"  ✓ Perfect prediction loss: {loss_perfect.item():.6f} (≈ 0)")

    # Gradient flow
    pred_depth.requires_grad_(True)
    pred_normal.requires_grad_(True)
    loss_d = masked_l1(pred_depth, target_depth, meta={"depth_mask": depth_mask})
    loss_n = cosine_loss(pred_normal, target_normal)
    (loss_d + loss_n).backward()
    assert pred_depth.grad is not None, "No gradient for depth"
    assert pred_normal.grad is not None, "No gradient for normals"
    print("  ✓ Gradient flow verified")

    print("\n" + "=" * 60)
    print("All checks PASSED.")
    print("=" * 60)
