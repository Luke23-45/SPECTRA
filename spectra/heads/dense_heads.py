"""
Dense prediction heads for pixel-wise multi-task learning (NYUv2).

These heads consume shared feature maps [B, D, H, W] from an encoder-decoder
backbone (SegNet) and produce per-pixel predictions for each task:
    - Semantic Segmentation: [B, num_classes, H, W]  → CrossEntropyLoss
    - Depth Estimation:      [B, 1, H, W]            → MaskedL1Loss
    - Surface Normals:       [B, 3, H, W]            → DenseCosineLoss

Architecture: 1×1 convolutions (per-pixel linear mixing)
    This is standard practice in dense prediction MTL (MTAN, PAD-Net, MTI-Net).
    The shared backbone does the heavy spatial reasoning; heads just project
    features to task-specific output spaces.

Why NOT larger kernels?
    The backbone already provides spatially-aware features. Adding 3×3 convs
    in heads would introduce task-specific spatial reasoning that the weighter
    cannot properly disentangle. 1×1 keeps the gradient signal clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DenseSegmentationHead(nn.Module):
    """
    Dense semantic segmentation head.

    Architecture: Conv1x1(D → D) → ReLU → Conv1x1(D → num_classes)

    Args:
        d_model: Input feature dimension from backbone.
        num_classes: Number of segmentation classes (13 for NYUv2).

    Input:  [B, D, H, W]
    Output: [B, num_classes, H, W] — raw logits (softmax applied by loss)
    """

    def __init__(self, d_model: int, num_classes: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        global_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, D, H, W] shared backbone features.
            global_ctx: Unused (API compatibility with scalar heads).

        Returns:
            [B, num_classes, H, W] raw logits.
        """
        return self.net(features)


class DenseRegressionHead(nn.Module):
    """
    Dense regression head for depth estimation or surface normal prediction.

    Architecture: Conv1x1(D → D) → ReLU → Conv1x1(D → output_dim)

    Args:
        d_model: Input feature dimension from backbone.
        output_dim: Number of output channels (1 for depth, 3 for normals).

    Input:  [B, D, H, W]
    Output: [B, output_dim, H, W]
    """

    def __init__(self, d_model: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, output_dim, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        global_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, D, H, W] shared backbone features.
            global_ctx: Unused (API compatibility with scalar heads).

        Returns:
            [B, output_dim, H, W] predictions.
        """
        return self.net(features)


# =============================================================================
# STANDALONE VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dense Heads — Shape Verification")
    print("=" * 60)

    B, D, H, W = 2, 64, 288, 384
    features = torch.randn(B, D, H, W)

    # Segmentation head (13 classes)
    seg_head = DenseSegmentationHead(d_model=D, num_classes=13)
    seg_out = seg_head(features)
    assert seg_out.shape == (B, 13, H, W), f"Seg shape: {seg_out.shape}"
    print(f"✓ Segmentation: {seg_out.shape}")

    # Depth head (1 channel)
    depth_head = DenseRegressionHead(d_model=D, output_dim=1)
    depth_out = depth_head(features)
    assert depth_out.shape == (B, 1, H, W), f"Depth shape: {depth_out.shape}"
    print(f"✓ Depth: {depth_out.shape}")

    # Normal head (3 channels)
    normal_head = DenseRegressionHead(d_model=D, output_dim=3)
    normal_out = normal_head(features)
    assert normal_out.shape == (B, 3, H, W), f"Normal shape: {normal_out.shape}"
    print(f"✓ Normals: {normal_out.shape}")

    print("=" * 60)
    print("All checks PASSED.")
    print("=" * 60)
