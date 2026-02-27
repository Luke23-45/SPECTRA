"""
spectra/core/gated_fusion.py
----------------------------
Primitive building blocks for the Asymmetric Latent Bottleneck (ALB).

Contains:
    - GatedResidualNetwork (GRN): TFT-style gated residual with SwiGLU
    - MultiScaleInceptionBlock: Multi-scale 1D/2D convolution feature extractor
    - VolatilityAwareGate: Gating mechanism that opens for high-frequency signals
    - SqueezeExcitation: Channel recalibration
    - SymmetryGate: Gated residual fusion

These primitives are composed by LateralBypass and ALB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. GATED RESIDUAL NETWORK (GRN)
# =====================================================================

class GatedResidualNetwork(nn.Module):
    """
    TFT-style Gated Residual Network with SiLU activation.

    Architecture: x → [W1 → SiLU → W2 → Dropout] * sigmoid(Gate(x)) + x → LayerNorm

    Used as the deep projection layer in the Expert branch of ALB.
    Stacking multiple GRNs encourages the expert to learn hierarchical
    high-frequency residuals.

    Args:
        d_model: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        
        out = self.w1(x_norm)
        out = F.silu(out)
        out = self.w2(out)
        out = self.dropout(out)
        
        g = torch.sigmoid(self.gate(x_norm))
        return residual + g * out


# =====================================================================
# 2. SQUEEZE-EXCITATION BLOCK
# =====================================================================

class SqueezeExcitation(nn.Module):
    """
    Channel recalibration via global average pooling + FC + sigmoid.

    Learns to emphasize informative channels and suppress less useful ones.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → [B, C, T] (channel-recalibrated)."""
        B, C, T = x.shape
        y = self.avg_pool(x).view(B, C)
        weights = self.fc(y).view(B, C, 1)
        return x * weights


# =====================================================================
# 3. SYMMETRY GATE
# =====================================================================

class SymmetryGate(nn.Module):
    """
    Gated residual fusion for multi-modal feature integration.

    g = sigmoid(Gate(x))
    out = LayerNorm(g * Proj(x) + (1 - g) * x)

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        g = torch.sigmoid(self.gate(x_norm))
        p = self.proj(x_norm)
        return (1 - g) * x + g * p


# =====================================================================
# 4. MULTI-SCALE INCEPTION BLOCK (1D)
# =====================================================================

class MultiScaleInceptionBlock(nn.Module):
    """
    Multi-scale 1D convolution for extracting features at different temporal resolutions.

    Four parallel paths:
        - Conv1d(k=3): local patterns (e.g., single-timestep spikes)
        - Conv1d(k=5): medium-range patterns (e.g., trends over 5 steps)
        - Conv1d(k=7): long-range patterns (e.g., slow drifts)
        - MaxPool(k=3) → Conv1d(k=1): peak detection

    All paths are concatenated, recalibrated via SE, fused through SymmetryGate,
    projected to output dimension, and added back as a residual.

    Args:
        in_dim: Input channels.
        out_dim: Output channels.
        dropout: Dropout probability.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = out_dim // 4

        # Multi-scale convolution paths
        # NOTE: Using GroupNorm(1) instead of BatchNorm1d because BN
        # crashes when T=1 (synthetic 2D data unsqueezed for ALB)
        self.branch_3 = nn.Sequential(
            nn.Conv1d(in_dim, mid, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid),
            nn.SiLU(),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv1d(in_dim, mid, kernel_size=5, padding=2),
            nn.GroupNorm(1, mid),
            nn.SiLU(),
        )
        self.branch_7 = nn.Sequential(
            nn.Conv1d(in_dim, mid, kernel_size=7, padding=3),
            nn.GroupNorm(1, mid),
            nn.SiLU(),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_dim, mid, kernel_size=1),
            nn.GroupNorm(1, mid),
            nn.SiLU(),
        )

        # Calibration and fusion
        self.se = SqueezeExcitation(mid * 4)
        self.gate = SymmetryGate(mid * 4)
        self.dropout = nn.Dropout1d(dropout)
        self.proj = nn.Conv1d(mid * 4, out_dim, kernel_size=1)

        # Residual connection (dimension matching)
        self.res = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T] input features in channel-first format.

        Returns:
            [B, C_out, T] multi-scale extracted features.
        """
        identity = self.res(x)

        # Parallel multi-scale paths
        out = torch.cat([
            self.branch_3(x),
            self.branch_5(x),
            self.branch_7(x),
            self.branch_pool(x),
        ], dim=1)  # [B, mid*4, T]

        # Squeeze-Excitation recalibration
        out = self.se(out)

        # SymmetryGate in [B, T, D] space
        out = out.transpose(1, 2)  # [B, T, mid*4]
        out = self.gate(out)
        out = out.transpose(1, 2)  # [B, mid*4, T]

        # Project and residual
        out = self.dropout(out)
        out = self.proj(out)  # [B, C_out, T]

        # Final: residual + normalize
        out = out.transpose(1, 2)  # [B, T, C_out]
        identity = identity.transpose(1, 2)  # [B, T, C_out]
        return self.norm(out + identity).transpose(1, 2)  # [B, C_out, T]


# =====================================================================
# 5. VOLATILITY-AWARE GATE
# =====================================================================

class VolatilityAwareGate(nn.Module):
    """
    Gating mechanism that automatically opens for high-frequency signals.

    This gate combines:
    1. Semantic gating: MLP on concatenated smooth/raw features
    2. Temporal volatility: |x_t - x_{t-1}| → per-channel projection

    The final gate value is:
        g = sigmoid(semantic(concat(smooth, raw)) + volatility_proj(delta))

    High-volatility timesteps (sudden spikes/drops) get higher gate values,
    allowing more bypass information through. Stable periods get lower values.

    Args:
        d_model: Hidden dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Per-channel volatility projection: maps scalar delta to d_model
        # so each feature dimension can have different volatility sensitivity
        self.volatility_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
        )

    def forward(
        self,
        smooth_ctx: torch.Tensor,
        raw_ctx: torch.Tensor,
        raw_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            smooth_ctx: [B, T, D] - Encoder (planner) output.
            raw_ctx: [B, T, D] - Bypass-processed features.
            raw_input: [B, T, C_in] - Raw input for volatility computation.

        Returns:
            gate: [B, T, D] gate values in [0, 1].
        """
        # Temporal volatility: |x_t - x_{t-1}|
        # [SOTA FIX]: Prevent T=1 degeneracy where |x_0 - 0| equals arbitrary magnitude.
        if raw_input.shape[1] == 1:
            delta = torch.zeros(raw_input.shape[0], 1, 1, device=raw_input.device)
        else:
            x_shifted = F.pad(raw_input[:, :-1, :], (0, 0, 1, 0))
            delta = (raw_input - x_shifted).abs().mean(dim=-1, keepdim=True)  # [B, T, 1]

        # Per-channel volatility embedding
        vol_embed = self.volatility_proj(delta)  # [B, T, D]

        # Semantic gate from feature concatenation
        combined = torch.cat([smooth_ctx, raw_ctx], dim=-1)  # [B, T, 2D]
        semantic_gate = self.gate_proj(combined)  # [B, T, D]

        # Combine: semantic + per-channel volatility
        g = torch.sigmoid(semantic_gate + vol_embed)
        return g
