"""
tests/test_modules_alb_robustness.py
------------------------------------
Comprehensive robustness tests for ALB and domain modules.

This test suite verifies:
- AsymmetricLatentBottleneck: Gradient divorce, manifold divergence
- SpatialALB: 2D spatial version
- LateralBypass: High-frequency extraction
- GatedResidualNetwork: TFT-style gating
- MultiScaleInceptionBlock: Multi-scale features
- VolatilityAwareGate: Semantic + volatility gating
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from spectra.core.alb import AsymmetricLatentBottleneck, SpatialALB
from spectra.core.lateral_bypass import LateralBypass
from spectra.core.gated_fusion import (
    GatedResidualNetwork,
    MultiScaleInceptionBlock,
    VolatilityAwareGate,
    SymmetryGate,
    SqueezeExcitation,
)


# ============================================================================
# CATEGORY A: CORRECTNESS TESTS - AsymmetricLatentBottleneck
# ============================================================================

class MockEncoder(nn.Module):
    """Simple mock encoder for testing."""
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, x, **kwargs):
        return self.net(x)


class TestALBCorrectness:
    """Verify AsymmetricLatentBottleneck produces correct outputs."""
    
    def test_alb_output_shapes_3d_input(self):
        """ALB should produce correct shapes for 3D input [B, T, C]."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
            n_expert_layers=2,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        assert out["planner"].shape == (B, T, d_model)
        assert out["planner_global"].shape == (B, d_model)
        assert out["expert"].shape == (B, T, d_model)
        assert out["expert_global"].shape == (B, d_model)
    
    def test_alb_output_shapes_2d_input(self):
        """ALB should handle 2D tabular input [B, C]."""
        input_dim, d_model = 20, 64
        B = 4
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, input_dim)
        out = alb(x)
        
        # Should auto-unsqueeze and squeeze back
        assert out["planner"].shape == (B, d_model)
        assert out["expert"].shape == (B, d_model)
    
    def test_alb_planner_global_is_mean_pooled(self):
        """planner_global should be mean of planner across time."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        expected_global = out["planner"].mean(dim=1)
        torch.testing.assert_close(out["planner_global"], expected_global, rtol=1e-5, atol=1e-6)
    
    def test_alb_expert_global_is_max_pooled(self):
        """expert_global should be max of expert across time."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        expected_global = out["expert"].max(dim=1)[0]
        torch.testing.assert_close(out["expert_global"], expected_global, rtol=1e-5, atol=1e-6)


class TestALBGradientDivorce:
    """Verify gradient divorce: expert gradients don't flow to encoder."""
    
    def test_alb_gradient_divorce_expert_path(self):
        """Expert loss should NOT backprop to encoder."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        # Backprop from expert only
        loss = out["expert"].sum()
        loss.backward()
        
        # Encoder should have ZERO gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert not encoder_has_grad, "GRADIENT DIVORCE FAILED: encoder received expert gradients!"
    
    def test_alb_gradient_flows_to_planner_path(self):
        """Planner loss SHOULD backprop to encoder."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        # Backprop from planner only
        loss = out["planner"].sum()
        loss.backward()
        
        # Encoder SHOULD have gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should receive planner gradients!"
    
    def test_alb_expert_proj_receives_gradients(self):
        """Expert projection should receive gradients from expert loss."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        loss = out["expert"].sum()
        loss.backward()
        
        # Expert projection should have gradients
        expert_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in alb.expert_proj.parameters()
        )
        assert expert_has_grad, "Expert projection should receive gradients!"


class TestALBManifoldDivergence:
    """Verify planner and expert produce different representations."""
    
    def test_alb_manifolds_differ(self):
        """Planner and expert should produce different representations."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        divergence = (out["planner"] - out["expert"]).abs().mean().item()
        assert divergence > 0.0, "MANIFOLD COLLAPSE: planner == expert!"


class TestALBMasking:
    """Verify masking support in ALB."""
    
    def test_alb_handles_padding_mask(self):
        """ALB should handle padding masks correctly."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -4:] = True  # Last 4 timesteps are padding
        
        out = alb(x, mask=mask)
        
        # Should not raise and produce valid outputs
        assert out["planner"].shape == (B, T, d_model)
        assert out["expert"].shape == (B, T, d_model)
    
    def test_alb_all_masked_global_expert(self):
        """When all positions masked, expert_global should be zeros."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        mask = torch.ones(B, T, dtype=torch.bool)  # All masked
        
        out = alb(x, mask=mask)
        
        # expert_global should be zeros for all-masked
        assert torch.allclose(out["expert_global"], torch.zeros(B, d_model), atol=1e-6)


# ============================================================================
# CATEGORY B: CORRECTNESS TESTS - SpatialALB
# ============================================================================

class MockEncoder2D(nn.Module):
    """Simple 2D encoder for testing."""
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)
    
    def forward(self, x, **kwargs):
        return self.conv(x)


class TestSpatialALBCorrectness:
    """Verify SpatialALB produces correct outputs."""
    
    def test_spatial_alb_output_shapes(self):
        """SpatialALB should produce correct shapes for 2D input."""
        in_channels, d_model = 3, 64
        B, H, W = 4, 32, 32
        
        encoder = MockEncoder2D(in_channels, d_model)
        spatial_alb = SpatialALB(encoder=encoder, d_model=d_model)
        
        x = torch.randn(B, in_channels, H, W)
        out = spatial_alb(x)
        
        assert out["planner"].shape == (B, d_model, H, W)
        assert out["expert"].shape == (B, d_model, H, W)
        assert "gate_mean" in out
    
    def test_spatial_alb_gradient_divorce(self):
        """SpatialALB should enforce gradient divorce."""
        in_channels, d_model = 3, 64
        B, H, W = 4, 32, 32
        
        encoder = MockEncoder2D(in_channels, d_model)
        spatial_alb = SpatialALB(encoder=encoder, d_model=d_model)
        
        x = torch.randn(B, in_channels, H, W)
        out = spatial_alb(x)
        
        # Backprop from expert
        loss = out["expert"].sum()
        loss.backward()
        
        # Encoder should NOT have gradients from expert path
        encoder_grad = encoder.conv.weight.grad
        assert encoder_grad is None or encoder_grad.abs().sum() == 0, \
            "SpatialALB gradient divorce failed!"
    
    def test_spatial_alb_manifolds_differ(self):
        """SpatialALB planner and expert should differ."""
        in_channels, d_model = 3, 64
        B, H, W = 4, 32, 32
        
        encoder = MockEncoder2D(in_channels, d_model)
        spatial_alb = SpatialALB(encoder=encoder, d_model=d_model)
        
        x = torch.randn(B, in_channels, H, W)
        out = spatial_alb(x)
        
        divergence = (out["planner"] - out["expert"]).abs().mean().item()
        assert divergence > 0.0, "SpatialALB manifold collapse!"


# ============================================================================
# CATEGORY C: CORRECTNESS TESTS - LateralBypass
# ============================================================================

class TestLateralBypassCorrectness:
    """Verify LateralBypass produces correct outputs."""
    
    def test_lateral_bypass_output_shape(self):
        """LateralBypass should preserve shape."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        bypass = LateralBypass(input_dim=input_dim, d_model=d_model)
        
        raw_input = torch.randn(B, T, input_dim)
        encoder_ctx = torch.randn(B, T, d_model)
        
        output = bypass(raw_input, encoder_ctx)
        
        assert output.shape == (B, T, d_model)
    
    def test_lateral_bypass_t1_handling(self):
        """LateralBypass should handle T=1 (tabular) inputs."""
        input_dim, d_model = 20, 64
        B = 4
        
        bypass = LateralBypass(input_dim=input_dim, d_model=d_model)
        
        raw_input = torch.randn(B, 1, input_dim)
        encoder_ctx = torch.randn(B, 1, d_model)
        
        output = bypass(raw_input, encoder_ctx)
        
        assert output.shape == (B, 1, d_model)
    
    def test_lateral_bypass_with_mask(self):
        """LateralBypass should handle padding masks."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        bypass = LateralBypass(input_dim=input_dim, d_model=d_model)
        
        raw_input = torch.randn(B, T, input_dim)
        encoder_ctx = torch.randn(B, T, d_model)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -4:] = True
        
        output = bypass(raw_input, encoder_ctx, mask=mask)
        
        assert output.shape == (B, T, d_model)


# ============================================================================
# CATEGORY D: CORRECTNESS TESTS - Gated Fusion Components
# ============================================================================

class TestGatedResidualNetwork:
    """Verify GatedResidualNetwork (GRN)."""
    
    def test_grn_output_shape(self):
        """GRN should preserve input shape."""
        d_model = 64
        B, T = 4, 16
        
        grn = GatedResidualNetwork(d_model=d_model)
        
        x = torch.randn(B, T, d_model)
        out = grn(x)
        
        assert out.shape == (B, T, d_model)
    
    def test_grn_residual_connection(self):
        """GRN should implement residual connection."""
        d_model = 64
        B, T = 4, 16
        
        grn = GatedResidualNetwork(d_model=d_model, dropout=0.0)
        grn.eval()  # Disable dropout for deterministic test
        
        x = torch.randn(B, T, d_model)
        out = grn(x)
        
        # Output should be x + gated_transform(norm(x))
        # Not just x (identity) and not completely different
        assert not torch.allclose(out, x, atol=1e-5)  # Should be transformed
    
    def test_grn_gradient_flows(self):
        """Gradient should flow through GRN."""
        d_model = 64
        B, T = 4, 16
        
        grn = GatedResidualNetwork(d_model=d_model)
        
        x = torch.randn(B, T, d_model, requires_grad=True)
        out = grn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestMultiScaleInceptionBlock:
    """Verify MultiScaleInceptionBlock."""
    
    def test_inception_output_shape(self):
        """Inception block should produce correct output shape."""
        in_dim, out_dim = 64, 64
        B, T = 4, 16
        
        block = MultiScaleInceptionBlock(in_dim=in_dim, out_dim=out_dim)
        
        x = torch.randn(B, in_dim, T)  # Channel-first
        out = block(x)
        
        assert out.shape == (B, out_dim, T)
    
    def test_inception_t1_bypass(self):
        """Inception block should handle T=1 with special bypass."""
        in_dim, out_dim = 64, 64
        B = 4
        
        block = MultiScaleInceptionBlock(in_dim=in_dim, out_dim=out_dim)
        
        x = torch.randn(B, in_dim, 1)  # T=1
        out = block(x)
        
        assert out.shape == (B, out_dim, 1)
        assert torch.isfinite(out).all()
    
    def test_inception_different_input_output_dims(self):
        """Inception block should handle dimension changes."""
        in_dim, out_dim = 32, 64
        B, T = 4, 16
        
        block = MultiScaleInceptionBlock(in_dim=in_dim, out_dim=out_dim)
        
        x = torch.randn(B, in_dim, T)
        out = block(x)
        
        assert out.shape == (B, out_dim, T)


class TestVolatilityAwareGate:
    """Verify VolatilityAwareGate."""
    
    def test_volatility_gate_output_shape(self):
        """Gate should produce correct output shape."""
        d_model = 64
        B, T = 4, 16
        
        gate = VolatilityAwareGate(d_model=d_model)
        
        smooth_ctx = torch.randn(B, T, d_model)
        raw_ctx = torch.randn(B, T, d_model)
        raw_input = torch.randn(B, T, 20)
        
        g = gate(smooth_ctx, raw_ctx, raw_input)
        
        assert g.shape == (B, T, d_model)
        assert (g >= 0).all() and (g <= 1).all()  # Sigmoid output
    
    def test_volatility_gate_t1_tabular(self):
        """Gate should handle T=1 tabular inputs."""
        d_model = 64
        B = 4
        
        gate = VolatilityAwareGate(d_model=d_model)
        
        smooth_ctx = torch.randn(B, 1, d_model)
        raw_ctx = torch.randn(B, 1, d_model)
        raw_input = torch.randn(B, 1, 20)
        
        g = gate(smooth_ctx, raw_ctx, raw_input)
        
        assert g.shape == (B, 1, d_model)
        assert torch.isfinite(g).all()


class TestSymmetryGate:
    """Verify SymmetryGate."""
    
    def test_symmetry_gate_output_shape(self):
        """SymmetryGate should preserve input shape."""
        dim = 64
        B, T = 4, 16
        
        gate = SymmetryGate(dim=dim)
        
        x = torch.randn(B, T, dim)
        out = gate(x)
        
        assert out.shape == (B, T, dim)


class TestSqueezeExcitation:
    """Verify SqueezeExcitation."""
    
    def test_se_output_shape(self):
        """SE should preserve input shape."""
        channels = 64
        B, C, T = 4, 64, 16
        
        se = SqueezeExcitation(channels=channels)
        
        x = torch.randn(B, C, T)
        out = se(x)
        
        assert out.shape == (B, C, T)
    
    def test_se_channel_recalibration(self):
        """SE should apply different weights to different channels."""
        channels = 64
        B, C, T = 4, 64, 16
        
        se = SqueezeExcitation(channels=channels)
        
        x = torch.randn(B, C, T)
        out = se(x)
        
        # Output should differ from input (channel recalibration)
        assert not torch.allclose(out, x, atol=1e-5)


# ============================================================================
# CATEGORY E: GRADIENT FLOW TESTS
# ============================================================================

class TestALBGradientFlow:
    """Verify gradient flow through ALB components."""
    
    def test_alb_full_gradient_flow(self):
        """Gradients should flow through ALB for planner path."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim, requires_grad=True)
        out = alb(x)
        
        # Backprop from planner_global
        loss = out["planner_global"].sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_lateral_bypass_gradient_flow(self):
        """Gradients should flow through LateralBypass."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        bypass = LateralBypass(input_dim=input_dim, d_model=d_model)
        
        raw_input = torch.randn(B, T, input_dim, requires_grad=True)
        encoder_ctx = torch.randn(B, T, d_model)  # Detached in practice
        
        output = bypass(raw_input, encoder_ctx)
        loss = output.sum()
        loss.backward()
        
        assert raw_input.grad is not None
        assert torch.isfinite(raw_input.grad).all()


# ============================================================================
# CATEGORY F: NUMERICAL STABILITY TESTS
# ============================================================================

class TestALBNumericalStability:
    """Verify numerical stability under extreme inputs."""
    
    def test_alb_large_input_values(self):
        """ALB should handle large input values."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim) * 100.0
        out = alb(x)
        
        assert torch.isfinite(out["planner"]).all()
        assert torch.isfinite(out["expert"]).all()
    
    def test_alb_small_input_values(self):
        """ALB should handle very small input values."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim) * 1e-6
        out = alb(x)
        
        assert torch.isfinite(out["planner"]).all()
        assert torch.isfinite(out["expert"]).all()
    
    def test_spatial_alb_large_values(self):
        """SpatialALB should handle large values."""
        in_channels, d_model = 3, 64
        B, H, W = 4, 32, 32
        
        encoder = MockEncoder2D(in_channels, d_model)
        spatial_alb = SpatialALB(encoder=encoder, d_model=d_model)
        
        x = torch.randn(B, in_channels, H, W) * 100.0
        out = spatial_alb(x)
        
        assert torch.isfinite(out["planner"]).all()
        assert torch.isfinite(out["expert"]).all()


# ============================================================================
# CATEGORY G: EDGE CASE TESTS
# ============================================================================

class TestALBEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_alb_single_timestep(self):
        """ALB should handle single timestep (T=1)."""
        input_dim, d_model = 20, 64
        B = 4
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, 1, input_dim)
        out = alb(x)
        
        assert out["planner"].shape == (B, 1, d_model)
        assert out["expert"].shape == (B, 1, d_model)
    
    def test_alb_single_sample(self):
        """ALB should handle batch size 1."""
        input_dim, d_model = 20, 64
        T = 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(1, T, input_dim)
        out = alb(x)
        
        assert out["planner"].shape == (1, T, d_model)
    
    def test_alb_long_sequence(self):
        """ALB should handle long sequences."""
        input_dim, d_model = 20, 64
        B, T = 2, 256
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        assert out["planner"].shape == (B, T, d_model)


# ============================================================================
# CATEGORY H: INVARIANT TESTS
# ============================================================================

class TestALBInvariants:
    """Test invariants that must hold across ALB components."""
    
    def test_all_alb_components_are_nn_modules(self):
        """All ALB components should be nn.Module instances."""
        assert isinstance(AsymmetricLatentBottleneck(MockEncoder(20, 64), 20, 64), nn.Module)
        assert isinstance(SpatialALB(MockEncoder2D(3, 64), 64), nn.Module)
        assert isinstance(LateralBypass(20, 64), nn.Module)
        assert isinstance(GatedResidualNetwork(64), nn.Module)
        assert isinstance(MultiScaleInceptionBlock(64, 64), nn.Module)
        assert isinstance(VolatilityAwareGate(64), nn.Module)
    
    def test_alb_caches_contexts(self):
        """ALB should cache contexts for spectral monitoring."""
        input_dim, d_model = 20, 64
        B, T = 4, 16
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, T, input_dim)
        out = alb(x)
        
        # Check caches exist
        assert hasattr(alb, 'last_planner_ctx')
        assert hasattr(alb, 'last_expert_ctx')
        assert hasattr(alb, 'last_planner_ctx_seq')
        assert hasattr(alb, 'last_expert_ctx_seq')


# ============================================================================
# CATEGORY I: REGRESSION TESTS
# ============================================================================

class TestALBRegressions:
    """Tests for specific bugs that have been fixed."""
    
    def test_alb_t1_attention_bypass(self):
        """
        REGRESSION: T=1 self-attention degenerates to identity.
        
        Fix: Bypass attention entirely for T=1 sequences.
        """
        input_dim, d_model = 20, 64
        B = 4
        
        encoder = MockEncoder(input_dim, d_model)
        alb = AsymmetricLatentBottleneck(
            encoder=encoder,
            input_dim=input_dim,
            d_model=d_model,
        )
        
        x = torch.randn(B, 1, input_dim)
        out = alb(x)
        
        # Should not raise and produce valid output
        assert torch.isfinite(out["expert"]).all()
    
    def test_inception_t1_thermal_death_fix(self):
        """
        REGRESSION: T=1 convolutions dilute energy with zeros.
        
        Fix: Use center weights with sqrt(kernel_size) gain scaling.
        """
        in_dim, out_dim = 64, 64
        B = 4
        
        block = MultiScaleInceptionBlock(in_dim=in_dim, out_dim=out_dim)
        
        x = torch.randn(B, in_dim, 1)
        out = block(x)
        
        # Output should not be near-zero (thermal death)
        assert out.abs().mean() > 0.01
    
    def test_volatility_gate_t1_energy_deviation_fix(self):
        """
        REGRESSION: T=1 temporal delta equals arbitrary magnitude.
        
        Fix: Use thermodynamic energy deviation instead.
        """
        d_model = 64
        B = 4
        
        gate = VolatilityAwareGate(d_model=d_model)
        
        smooth_ctx = torch.randn(B, 1, d_model)
        raw_ctx = torch.randn(B, 1, d_model)
        raw_input = torch.randn(B, 1, 20)
        
        g = gate(smooth_ctx, raw_ctx, raw_input)
        
        # Gate should be finite and in valid range
        assert torch.isfinite(g).all()
        assert (g >= 0).all() and (g <= 1).all()


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
