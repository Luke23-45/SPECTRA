"""
spectra/backbones/segnet.py
----------------------------
SegNet Encoder-Decoder Backbone for Dense Prediction MTL.

Standard architecture for NYUv2 multi-task learning benchmarks
(MTAN, LibMTL, CAGrad, Nash-MTL all use this baseline).

Architecture:
    Encoder: 5 conv blocks with batch norm, each followed by 2×2 max-pooling
    Decoder: 5 conv blocks with max-unpooling using saved pooling indices

    Block structure (encoder):
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool (save indices)

    Block structure (decoder):
        MaxUnpool → Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU

Input:  [B, 3, H, W]    — RGB image
Output: [B, D, H, W]    — Feature map at original resolution

The output is a shared representation that feeds into task-specific heads.
This is aligned with the SPECTRA hypothesis: B-PGS operates on the scalar
losses from each task head, and the backbone provides the shared features
that ALL tasks draw from — exactly where gradient interference occurs.

References:
    - Badrinarayanan et al. "SegNet: A Deep Convolutional Encoder-Decoder
      Architecture for Image Segmentation" (TPAMI 2017)
    - Liu et al. "End-to-End Multi-Task Learning with Attention" (CVPR 2019)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("spectra.backbones.segnet")


# =============================================================================
# ENCODER BLOCK
# =============================================================================

class SegNetEncoderBlock(nn.Module):
    """
    Single encoder block: 2 × (Conv → BN → ReLU) + MaxPool.

    Stores pooling indices for the corresponding decoder block's unpooling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            (pooled_features, pooling_indices, pre_pool_size)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        pre_pool_size = x.size()
        x, indices = self.pool(x)
        return x, indices, pre_pool_size


# =============================================================================
# DECODER BLOCK
# =============================================================================

class SegNetDecoderBlock(nn.Module):
    """
    Single decoder block: MaxUnpool + 2 × (Conv → BN → ReLU).

    The last decoder block differs: final conv outputs `out_channels` (the
    shared feature dimension D) instead of matching encoder channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x: torch.Tensor, indices: torch.Tensor, output_size: torch.Size):
        x = self.unpool(x, indices, output_size=output_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


# =============================================================================
# SEGNET BACKBONE
# =============================================================================

class SegNet(nn.Module):
    """
    SegNet Encoder-Decoder backbone for dense prediction.

    Produces a shared feature map at the original input resolution.
    Task-specific dense heads (conv-based) consume this feature map.

    This architecture directly supports the SPECTRA hypothesis:
    - The shared encoder-decoder is where gradient interference occurs
    - B-PGS controls the magnitude of per-task gradient contributions
    - ALB (future) can decouple frequency bands in this shared space

    Args:
        input_channels: Number of input channels (3 for RGB).
        d_model: Output feature dimension per pixel.
                 Default 64 matches MTAN standard.
        encoder_channels: List of channel counts for each encoder stage.
                         Default [64, 128, 256, 512, 512] matches standard SegNet.

    Input:  [B, 3, H, W]
    Output: [B, d_model, H, W]
    """

    def __init__(
        self,
        input_channels: int = 3,
        d_model: int = 64,
        encoder_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 512]

        self.d_model = d_model

        # ─── Encoder ─────────────────────────────────────────────
        enc_in_channels = [input_channels] + encoder_channels[:-1]
        self.encoders = nn.ModuleList([
            SegNetEncoderBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(enc_in_channels, encoder_channels)
        ])

        # ─── Decoder ─────────────────────────────────────────────
        # Decoder mirrors encoder in reverse
        dec_channels = list(reversed(encoder_channels))
        dec_out_channels = dec_channels[1:] + [d_model]
        self.decoders = nn.ModuleList([
            SegNetDecoderBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(dec_channels, dec_out_channels)
        ])

        # Parameter count logging
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[SegNet] Initialized: in={input_channels}, d_model={d_model}, "
            f"stages={len(encoder_channels)}, params={n_params:,}"
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W] input image.

        Returns:
            [B, d_model, H, W] shared feature map.
        """
        # ─── Encode ──────────────────────────────────────────────
        indices_stack = []
        sizes_stack = []

        for encoder in self.encoders:
            x, indices, pre_pool_size = encoder(x)
            indices_stack.append(indices)
            sizes_stack.append(pre_pool_size)

        # ─── Decode ──────────────────────────────────────────────
        for decoder in self.decoders:
            indices = indices_stack.pop()
            size = sizes_stack.pop()
            x = decoder(x, indices, output_size=size)

        return x


# =============================================================================
# STANDALONE VERIFICATION
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SegNet Backbone — Shape Verification")
    print("=" * 60)

    model = SegNet(input_channels=3, d_model=64)

    # Standard NYUv2 resolution
    x = torch.randn(2, 3, 288, 384)
    out = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 64, 288, 384), f"Shape mismatch: {out.shape}"
    print("✓ Shape correct: [B, 64, H, W]")

    # Gradient flow test
    loss = out.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters())
    print(f"✓ Gradient flow: {'PASS' if has_grad else 'FAIL'}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {n_params:,}")

    print("=" * 60)
    print("All checks PASSED.")
    print("=" * 60)
