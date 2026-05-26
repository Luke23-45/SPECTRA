"""
Lateral Expert Bypass: High-frequency feature extraction path.

Processes raw input through multi-scale convolutions and fuses with
(detached) encoder context via volatility-aware gating.

This is the "sharp" pathway in the Asymmetric Latent Bottleneck that
captures high-frequency discriminative signals that the smooth
Transformer trunk suppresses due to NTK spectral bias.

Architecture:
    raw_input → [Linear Proj] → [Inception Block] → bypass_features
    bypass_features + detached_encoder → [VolatilityGate] → gated
    output = detached_encoder + gate * bypass_features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from spectra.core.gated_fusion import (
    MultiScaleInceptionBlock,
    SymmetryGate,
    VolatilityAwareGate,
)


class LateralBypass(nn.Module):
    """
    Lateral Expert Bypass for high-frequency feature extraction.

    Computes a "sharp" representation by:
    1. Projecting raw input to hidden dimension
    2. Extracting multi-scale temporal features via inception convolutions
    3. Gating fusion with encoder context based on temporal volatility

    The output has the same shape as the encoder context, enabling
    clean residual addition in the ALB.

    Args:
        input_dim: Raw input feature dimension (e.g., 3 for RGB, 28 for clinical).
        d_model: Hidden dimension (must match encoder output dim).
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project raw input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

        # Multi-modal fusion gate
        self.group_gate = SymmetryGate(d_model)

        # Multi-scale feature extraction
        self.feat_extractor = MultiScaleInceptionBlock(d_model, d_model, dropout=dropout)

        # Volatility-aware gating
        self.gate = VolatilityAwareGate(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        raw_input: torch.Tensor,
        encoder_ctx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extracts high-frequency features and fuses with encoder context.

        Args:
            raw_input: [B, T, C_in] raw input features.
            encoder_ctx: [B, T, D] encoder output (should be DETACHED from planner).
            mask: [B, T] optional padding mask (True = pad, False = valid).

        Returns:
            [B, T, D] expert context with high-frequency bypass information.
        """
        B, T, C = raw_input.shape

        # 1. Project raw input to hidden dimension
        z_raw = self.input_proj(raw_input)  # [B, T, D]

        # 2. Gated multi-modal fusion
        z_raw = self.group_gate(z_raw)  # [B, T, D]

        # 3. Multi-scale inception extraction (needs channel-first)
        z_inception = self.feat_extractor(z_raw.transpose(1, 2))  # [B, D, T]
        z_bypass = z_inception.transpose(1, 2)  # [B, T, D]
        z_bypass = self.dropout(z_bypass)

        # 4. Apply mask to bypass features
        if mask is not None:
            z_bypass = z_bypass.masked_fill(mask.unsqueeze(-1), 0.0)

        # 5. Volatility-aware gating
        gate_values = self.gate(encoder_ctx, z_bypass, raw_input)  # [B, T, D]

        # 6. Expert manifold fusion: encoder + gated bypass
        output = encoder_ctx + gate_values * z_bypass

        # 7. Apply mask to output
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)

        return output

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, d_model={self.d_model}"
