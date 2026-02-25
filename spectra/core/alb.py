"""
spectra/core/alb.py
-------------------
Asymmetric Latent Bottleneck (ALB): Spectral Decoupler for Multi-Task Learning.

Core Contribution #2 of the SPECTRA framework.

Theoretical Foundation:
    Neural Tangent Kernel (NTK) spectral bias causes shared trunks to converge
    via large eigenvalues first (low-frequency smooth features), systematically
    under-representing high-frequency discriminative boundaries. (F-Principle;
    NTKMTL, Qin et al. 2025).

    ALB architecturally enforces spectral separation:
    - Main trunk (Planner): learns low-freq smooth manifolds (for generative tasks)
    - Gated lateral bypass (Expert): injects high-freq sharp features (for classification)
    - Gradient Divorce: .detach() prevents expert gradients from polluting the planner

Architecture:
    Input ──┬──→ [Shared Encoder] ──→ ctx_planner (Smooth, Low-Freq)
            │                            │ .detach()
            └──→ [LateralBypass] ──→ [Self-Attn] ──→ [GRN Stack] ──→ ctx_expert (Sharp, High-Freq)

References:
    - Qin et al., "NTKMTL" (arXiv Oct 2025) — NTK spectral task balance
    - F-Principle / frequency bias in neural networks
    - APEX-MoE ALB (our prior clinical work, generalized here)
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from spectra.core.gated_fusion import GatedResidualNetwork
from spectra.core.lateral_bypass import LateralBypass

logger = logging.getLogger("spectra.alb")


class AsymmetricLatentBottleneck(nn.Module):
    """
    Asymmetric Latent Bottleneck: Spectral Decoupler for MTL.

    Produces two distinct representation manifolds from a shared encoder:
        1. ctx_planner: Low-frequency smooth features (generative tasks)
        2. ctx_expert: High-frequency sharp features (discriminative tasks)

    The expert manifold receives DETACHED planner context (gradient divorce)
    to prevent discriminative gradients from corrupting the generative trunk.

    Args:
        encoder: Shared backbone encoder. Must accept input [B, T, C_in] and
                 return features [B, T, D].
        input_dim: Raw input feature dimension.
        d_model: Hidden dimension (must match encoder output).
        n_expert_layers: Number of GRN layers in expert projection (depth).
        n_heads: Number of attention heads for expert self-attention.
        dropout: Dropout probability.
        init_mode: Weight initialization for expert branch.
                   'orthogonal': preserves variance, encourages high-freq residuals.
                   'default': standard PyTorch initialization.
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        d_model: int,
        n_expert_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        init_mode: str = "orthogonal",
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model

        # Expert feature extraction via lateral bypass
        self.lateral_bypass = LateralBypass(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )

        # Expert self-organization via self-attention
        self.expert_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Expert deep projection (GRN stack)
        self.expert_proj = nn.Sequential(
            *[GatedResidualNetwork(d_model, dropout=dropout) for _ in range(n_expert_layers)]
        )
        
        # Final expert norm (required to normalize after Pre-Norm stack)
        self.expert_final_norm = nn.LayerNorm(d_model)

        # Initialize expert weights
        if init_mode == "orthogonal":
            self._init_expert_weights()

    def _init_expert_weights(self) -> None:
        """
        Orthogonal initialization with gain > 1.0 for expert branch.

        Rationale (from v3.md): Standard random initialization is spectrally
        white or low-freq biased. Orthogonal init with higher gain encourages
        the expert to learn high-frequency residuals rather than redundant
        low-frequency features from the trunk.
        """
        for module in [self.expert_proj, self.lateral_bypass, self.expert_self_attn]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    nn.init.orthogonal_(m.weight, gain=1.2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.MultiheadAttention):
                    # For MHA, the in_proj_weight concatenates Q, K, V
                    if m.in_proj_weight is not None:
                        nn.init.orthogonal_(m.in_proj_weight, gain=1.2)
                    if m.in_proj_bias is not None:
                        nn.init.zeros_(m.in_proj_bias)
                    if m.out_proj.weight is not None:
                        nn.init.orthogonal_(m.out_proj.weight, gain=1.2)
                    if m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **encoder_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Produces decoupled planner (low-freq) and expert (high-freq) representations.

        Args:
            x: [B, T, C_in] raw input features.
            mask: [B, T] optional padding mask (True = pad, False = valid).
            **encoder_kwargs: Additional arguments passed to the shared encoder.

        Returns:
            Dict with:
                "ctx_planner": [B, T, D]  — Low-freq features for generative heads
                "global_planner": [B, D]  — Pooled planner summary (mean pool)
                "ctx_expert": [B, T, D]   — High-freq features for discriminative heads
                "global_expert": [B, D]   — Pooled expert summary (max pool)
        """
        # 1. Shared Encoding → Planner Manifold
        ctx_planner = self.encoder(x, **encoder_kwargs)  # [B, T, D]

        # 2. Planner Global Summary (masked mean pooling)
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()  # [B, T, 1]
            global_planner = (ctx_planner * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            global_planner = ctx_planner.mean(dim=1)  # [B, D]

        # 3. GRADIENT DIVORCE — CRITICAL
        # Detach planner context before feeding to expert branch.
        # This prevents discriminative task gradients from corrupting
        # the generative planner's smooth manifold.
        ctx_detached = ctx_planner.detach()

        # 4. Expert Feature Extraction via Lateral Bypass
        ctx_bypass = self.lateral_bypass(x, ctx_detached, mask=mask)  # [B, T, D]

        # 5. Expert Self-Organization (Self-Attention)
        # The expert attends to its own features to discover correlations
        # in the high-frequency residual space.
        # Switched to Pre-Norm for Training Stability (LLaMA/GPT-3 style)
        ctx_normed = self.attn_norm(ctx_bypass)
        key_padding_mask = mask if mask is not None else None
        attn_out, _ = self.expert_self_attn(
            ctx_normed, ctx_normed, ctx_normed,
            key_padding_mask=key_padding_mask,
        )
        ctx_expert = ctx_bypass + self.attn_dropout(attn_out)

        # 6. Expert Deep Projection (GRN Stack)
        ctx_expert = self.expert_proj(ctx_expert)  # [B, T, D]
        
        # 6.5 Final normalization (required since GRN is now Pre-Norm)
        ctx_expert = self.expert_final_norm(ctx_expert)

        # 7. Apply mask
        if mask is not None:
            ctx_expert = ctx_expert.masked_fill(mask.unsqueeze(-1), 0.0)

        # 8. Expert Global Summary (max pooling for peak/anomaly detection)
        # Max pooling captures the most extreme activations — ideal for
        # detecting anomalies and sharp decision boundaries.
        if mask is not None:
            # Replace padded positions with -inf for max pool
            ctx_for_max = ctx_expert.masked_fill(mask.unsqueeze(-1), float("-inf"))
            global_expert = ctx_for_max.max(dim=1)[0]
            # Handle all-masked edge case: [B, 1] → broadcast to [B, D]
            all_masked = mask.all(dim=1).unsqueeze(-1)  # [B, 1]
            global_expert = torch.where(all_masked, torch.zeros_like(global_expert), global_expert)
        else:
            global_expert = ctx_expert.max(dim=1)[0]  # [B, D]

        return {
            "ctx_planner": ctx_planner,
            "global_planner": global_planner,
            "ctx_expert": ctx_expert,
            "global_expert": global_expert,
        }

    def extra_repr(self) -> str:
        n_grn = len(self.expert_proj)
        return f"d_model={self.d_model}, expert_layers={n_grn}"


class SpatialALB(nn.Module):
    """
    Spatial Asymmetric Latent Bottleneck (ALB2D).
    Role: Spectral Decoupler for Spatial Manifolds (NYUv2/SegNet).

    Unlike the 1D variant (for temporal/tabular data), SpatialALB uses
    2D convolutions to preserve spatial topology while enforcing
    gradient divorce between the planner and expert branches.

    Mechanisms:
      1. Gradient Divorce: Expert branch receives .detach() trunk context.
      2. Volatility Gate (2D): 1x1 Conv + Sigmoid detects high-variance pixels (edges).
      3. Global Context: Expert uses dilated convolutions to capture wide-range
         spatial context without pooling, preserving sharp resolution.
    """

    def __init__(self, encoder: nn.Module, d_model: int, expert_gain: float = 1.2):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model

        # 1. Volatility Gate (2D): Detects sharp feature areas (edges)
        self.gate = nn.Sequential(
            nn.Conv2d(d_model, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 2. Expert Path (High-Freq): Dilated context extraction
        self.expert = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
        )

        # 3. Orthogonal Initialization for Expert Stability
        self._init_weights(expert_gain)

    def _init_weights(self, gain: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **encoder_kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Raw input tensor [B, C, H, W].
            **encoder_kwargs: Arguments for the internal encoder.

        Returns:
            Dict containing planner and expert manifolds.
        """
        # 1. Primary Encoding Path (Smooth Trunk)
        x_trunk = self.encoder(x, **encoder_kwargs)

        # CRITICAL: Gradient Divorce
        x_detached = x_trunk.detach()

        # 2. Compute Spatial Gating Map
        gate_map = self.gate(x_detached)  # [B, 1, H, W]

        # 3. Extract High-Freq Residuals
        expert_res = self.expert(x_detached)  # [B, D, H, W]

        # 4. Injection: Trunk + (Gate * Expert_Residual)
        ctx_expert = x_trunk + (gate_map * expert_res)

        return {
            "ctx_planner": x_trunk,
            "ctx_expert": ctx_expert,
            "gate_mean": gate_map.mean(),
        }

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


# =====================================================================
# STANDALONE VERIFICATION
# =====================================================================

if __name__ == "__main__":
    """Quick sanity check — run with: python -m spectra.core.alb"""
    print("=" * 60)
    print("ALB (Asymmetric Latent Bottleneck) — Standalone Verification")
    print("=" * 60)

    torch.manual_seed(42)

    # Mock encoder: simple MLP trunk
    class MockEncoder(nn.Module):
        def __init__(self, in_dim, d_model):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        def forward(self, x, **kwargs):
            return self.net(x)

    input_dim = 20
    d_model = 64
    B, T = 8, 16

    encoder = MockEncoder(input_dim, d_model)
    alb = AsymmetricLatentBottleneck(
        encoder=encoder,
        input_dim=input_dim,
        d_model=d_model,
        n_expert_layers=3,
        n_heads=8,
        dropout=0.1,
        init_mode="orthogonal",
    )

    x = torch.randn(B, T, input_dim)
    out = alb(x)

    print(f"\nInput shape: {x.shape}")
    print(f"ctx_planner shape: {out['ctx_planner'].shape}")
    print(f"global_planner shape: {out['global_planner'].shape}")
    print(f"ctx_expert shape: {out['ctx_expert'].shape}")
    print(f"global_expert shape: {out['global_expert'].shape}")

    # Shape checks
    assert out["ctx_planner"].shape == (B, T, d_model)
    assert out["global_planner"].shape == (B, d_model)
    assert out["ctx_expert"].shape == (B, T, d_model)
    assert out["global_expert"].shape == (B, d_model)
    print("✓ All shapes correct")

    # Gradient divorce check
    loss_expert = out["ctx_expert"].sum()
    loss_expert.backward()

    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in encoder.parameters()
    )
    assert not encoder_has_grad, "GRADIENT DIVORCE FAILED: encoder received expert gradients!"
    print("✓ Gradient divorce verified: encoder has ZERO expert gradients")

    expert_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in alb.expert_proj.parameters()
    )
    assert expert_has_grad, "GRADIENT FAILURE: expert_proj has no gradients!"
    print("✓ Expert projection receives gradients")

    # Representational divergence check
    divergence = (out["ctx_planner"] - out["ctx_expert"]).abs().mean().item()
    print(f"✓ Representational divergence: {divergence:.4f} (>0 means manifolds differ)")
    assert divergence > 0.0, "MANIFOLD COLLAPSE: planner == expert!"

    print(f"\n{'=' * 60}")
    print("ALB: All verifications PASSED.")
    print(f"{'=' * 60}")

    # =====================================================================
    # SPATIAL ALB (2D) VERIFICATION
    # =====================================================================
    print("\n" + "=" * 60)
    print("SpatialALB (2D) — Standalone Verification")
    print("=" * 60)

    B, C, H, W = 4, 3, 32, 32
    D = 64
    mock_encoder_2d = nn.Conv2d(C, D, kernel_size=3, padding=1)
    spatial_alb = SpatialALB(encoder=mock_encoder_2d, d_model=D)
    
    x_img = torch.randn(B, C, H, W, requires_grad=True)
    out_2d = spatial_alb(x_img)

    print(f"\nInput shape: {x_img.shape}")
    print(f"ctx_planner shape: {out_2d['ctx_planner'].shape}")
    print(f"ctx_expert shape: {out_2d['ctx_expert'].shape}")
    print(f"gate_mean: {out_2d['gate_mean'].item():.4f}")

    # 1. Shape Checks
    assert out_2d["ctx_planner"].shape == (B, D, H, W)
    assert out_2d["ctx_expert"].shape == (B, D, H, W)
    print("✓ 2D Shapes correct")

    # 2. Gradient Divorce Check
    loss_expert = out_2d["ctx_expert"].sum()
    loss_expert.backward(retain_graph=True)

    # The encoder should NOT receive grads from the expert residual path
    encoder_grad_sum = mock_encoder_2d.weight.grad.abs().sum().item()
    print(f"Encoder grad sum: {encoder_grad_sum:.4f}")
    
    # Reset and check planner path (should have grads)
    mock_encoder_2d.weight.grad.zero_()
    loss_planner = out_2d["ctx_planner"].sum()
    loss_planner.backward()
    planner_grad_sum = mock_encoder_2d.weight.grad.abs().sum().item()
    print(f"Planner grad sum: {planner_grad_sum:.4f}")
    assert planner_grad_sum > 0, "GRADIENT FAILURE: Encoder has no grads from planner!"

    # 3. Manifold Divergence
    divergence_2d = (out_2d["ctx_planner"] - out_2d["ctx_expert"]).abs().mean().item()
    print(f"✓ Representational divergence (2D): {divergence_2d:.4f}")
    assert divergence_2d > 0.0, "MANIFOLD COLLAPSE in 2D!"

    print(f"\n{'=' * 60}")
    print("SpatialALB (2D): All verifications PASSED.")
    print(f"{'=' * 60}")
