"""
Generic Shared Trunk backbone for non-vision benchmarks (synthetic, clinical).

Simple MLP with residual connections and layer normalization.
Outputs [B, T, D] features compatible with ALB and task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SharedTrunk(nn.Module):
    """
    Generic MLP trunk for tabular / sequence data.

    Architecture:
        Input → Linear → LayerNorm → SiLU → (ResBlock × N) → Output

    Each ResBlock:
        x → Linear → SiLU → Linear → Dropout → + x → LayerNorm

    Args:
        input_dim: Raw input feature dimension.
        d_model: Internal hidden dimension (output dimension).
        n_layers: Number of residual blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList([
            ResBlock(d_model, dropout) for _ in range(n_layers)
        ])
        
        # Required for Pre-Norm architecture to ensure final features are normalized
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, D_in] or [B, T, D_in] input features.

        Returns:
            [B, D_model] or [B, T, D_model] hidden features.
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.final_norm(h)


class ResBlock(nn.Module):
    """Residual block with pre-norm."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm architecture: x + f(norm(x))
        # Eliminates the need for learning rate warmup and prevents exploding gradients
        return x + self.net(self.norm(x))
