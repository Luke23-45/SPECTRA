"""
spectra/heads/regression.py & classification.py — Generic task heads.
spectra/heads/__init__.py already exists.

This file provides simple heads for synthetic/clinical benchmarks.
NYUv2 dense heads (segmentation, depth, normals) will be separate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RegressionHead(nn.Module):
    """
    Simple regression prediction head.

    Architecture: Linear(d_model → d_model // 2) → SiLU → Linear(→ output_dim)

    Args:
        d_model: Input feature dimension.
        output_dim: Number of output values (default: 1).
    """

    def __init__(self, d_model: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, features: torch.Tensor, global_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, D] or [B, T, D] features from backbone/ALB.
            global_ctx: [B, D] optional global context (unused by default).

        Returns:
            [B, output_dim] predictions.
        """
        if features.dim() == 3:
            features = features.mean(dim=1)  # Pool temporal dim
        return self.net(features)


class ClassificationHead(nn.Module):
    """
    Binary or multi-class classification head.

    Architecture: Linear(d_model → d_model // 2) → SiLU → Dropout → Linear(→ num_classes)

    Args:
        d_model: Input feature dimension.
        num_classes: 1 for binary (sigmoid), >1 for multi-class (softmax).
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, features: torch.Tensor, global_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.net(features)  # Raw logits (loss applies sigmoid/softmax)
