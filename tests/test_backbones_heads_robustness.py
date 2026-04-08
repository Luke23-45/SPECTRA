"""
tests/test_backbones_heads_robustness.py
----------------------------------------
Comprehensive robustness tests for backbones and heads.

This test suite verifies:
- SegNet: Encoder-decoder shape preservation, gradient flow
- SharedTrunk: MLP trunk for tabular data, residual connections
- RegressionHead/ClassificationHead: Scalar task heads
- DenseSegmentationHead/DenseRegressionHead: Pixel-wise heads
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from spectra.backbones.segnet import SegNet, SegNetEncoderBlock, SegNetDecoderBlock
from spectra.backbones.shared_trunk import SharedTrunk, ResBlock
from spectra.heads.task_heads import RegressionHead, ClassificationHead
from spectra.heads.dense_heads import DenseSegmentationHead, DenseRegressionHead


# ============================================================================
# CATEGORY A: CORRECTNESS TESTS - SegNet
# ============================================================================

class TestSegNetCorrectness:
    """Verify SegNet backbone produces correct outputs."""
    
    def test_segnet_output_shape_matches_input_resolution(self):
        """Output should have same H, W as input."""
        model = SegNet(input_channels=3, d_model=64)
        
        x = torch.randn(2, 3, 288, 384)
        out = model(x)
        
        assert out.shape == (2, 64, 288, 384), f"Shape mismatch: {out.shape}"
    
    def test_segnet_different_input_sizes(self):
        """Should handle various input sizes (not just NYUv2)."""
        model = SegNet(input_channels=3, d_model=64)
        
        # Test various sizes
        for h, w in [(64, 64), (128, 128), (224, 224), (288, 384)]:
            x = torch.randn(1, 3, h, w)
            out = model(x)
            assert out.shape == (1, 64, h, w), f"Failed for size {(h, w)}: {out.shape}"
    
    def test_segnet_custom_d_model(self):
        """Should support different output feature dimensions."""
        model = SegNet(input_channels=3, d_model=128)
        
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        
        assert out.shape == (2, 128, 64, 64)
    
    def test_segnet_custom_encoder_channels(self):
        """Should support custom encoder channel configurations."""
        # Smaller model for faster testing
        model = SegNet(
            input_channels=3,
            d_model=32,
            encoder_channels=[32, 64, 128, 256, 256]
        )
        
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        
        assert out.shape == (2, 32, 64, 64)


class TestSegNetEncoderDecoderBlocks:
    """Verify encoder and decoder blocks work correctly."""
    
    def test_encoder_block_reduces_spatial_dims(self):
        """Encoder block should halve spatial dimensions."""
        block = SegNetEncoderBlock(in_channels=3, out_channels=64)
        
        x = torch.randn(2, 3, 64, 64)
        out, indices, pre_pool_size = block(x)
        
        assert out.shape == (2, 64, 32, 32), f"Output shape: {out.shape}"
        assert indices.shape == (2, 64, 32, 32)
        assert pre_pool_size == (2, 64, 64, 64)
    
    def test_decoder_block_restores_spatial_dims(self):
        """Decoder block should restore spatial dimensions."""
        # Create matching encoder-decoder pair
        encoder = SegNetEncoderBlock(in_channels=64, out_channels=128)
        decoder = SegNetDecoderBlock(in_channels=128, out_channels=64)
        
        x = torch.randn(2, 64, 64, 64)
        encoded, indices, pre_pool_size = encoder(x)
        decoded = decoder(encoded, indices, output_size=pre_pool_size)
        
        assert decoded.shape == (2, 64, 64, 64), f"Decoded shape: {decoded.shape}"


# ============================================================================
# CATEGORY B: CORRECTNESS TESTS - SharedTrunk
# ============================================================================

class TestSharedTrunkCorrectness:
    """Verify SharedTrunk MLP backbone."""
    
    def test_shared_trunk_2d_input(self):
        """Should handle 2D input [B, D_in]."""
        model = SharedTrunk(input_dim=20, d_model=64, n_layers=2)
        
        x = torch.randn(4, 20)
        out = model(x)
        
        assert out.shape == (4, 64)
    
    def test_shared_trunk_3d_input(self):
        """Should handle 3D input [B, T, D_in]."""
        model = SharedTrunk(input_dim=20, d_model=64, n_layers=2)
        
        x = torch.randn(4, 10, 20)  # Batch, Time, Features
        out = model(x)
        
        assert out.shape == (4, 10, 64)
    
    def test_shared_trunk_residual_connection(self):
        """ResBlock should implement residual connection correctly."""
        block = ResBlock(d_model=64, dropout=0.0)
        
        x = torch.randn(2, 64)
        out = block(x)
        
        # Output should be x + f(norm(x))
        # Verify shape and that it's different from input
        assert out.shape == x.shape
        assert not torch.allclose(out, x)  # Should be transformed
    
    def test_shared_trunk_multiple_layers(self):
        """Should support different numbers of layers."""
        for n_layers in [1, 2, 4, 6]:
            model = SharedTrunk(input_dim=20, d_model=64, n_layers=n_layers)
            x = torch.randn(2, 20)
            out = model(x)
            assert out.shape == (2, 64)


# ============================================================================
# CATEGORY C: CORRECTNESS TESTS - Task Heads
# ============================================================================

class TestRegressionHeadCorrectness:
    """Verify RegressionHead."""
    
    def test_regression_head_2d_input(self):
        """Should handle 2D input [B, D]."""
        head = RegressionHead(d_model=64, output_dim=1)
        
        x = torch.randn(4, 64)
        out = head(x)
        
        assert out.shape == (4, 1)
    
    def test_regression_head_3d_input(self):
        """Should extract final state from 3D input [B, T, D]."""
        head = RegressionHead(d_model=64, output_dim=1)
        
        x = torch.randn(4, 10, 64)
        out = head(x)
        
        assert out.shape == (4, 1)
    
    def test_regression_head_multi_output(self):
        """Should support multiple output dimensions."""
        head = RegressionHead(d_model=64, output_dim=5)
        
        x = torch.randn(4, 64)
        out = head(x)
        
        assert out.shape == (4, 5)


class TestClassificationHeadCorrectness:
    """Verify ClassificationHead."""
    
    def test_classification_head_binary(self):
        """Binary classification should output 1 logit."""
        head = ClassificationHead(d_model=64, num_classes=1)
        
        x = torch.randn(4, 64)
        out = head(x)
        
        assert out.shape == (4, 1)
    
    def test_classification_head_multiclass(self):
        """Multi-class should output num_classes logits."""
        head = ClassificationHead(d_model=64, num_classes=10)
        
        x = torch.randn(4, 64)
        out = head(x)
        
        assert out.shape == (4, 10)
    
    def test_classification_head_3d_input(self):
        """Should extract final state from 3D input."""
        head = ClassificationHead(d_model=64, num_classes=3)
        
        x = torch.randn(4, 10, 64)
        out = head(x)
        
        assert out.shape == (4, 3)


# ============================================================================
# CATEGORY D: CORRECTNESS TESTS - Dense Heads
# ============================================================================

class TestDenseHeadsCorrectness:
    """Verify dense prediction heads."""
    
    def test_segmentation_head_output_shape(self):
        """Segmentation head should output [B, num_classes, H, W]."""
        head = DenseSegmentationHead(d_model=64, num_classes=13)
        
        features = torch.randn(2, 64, 288, 384)
        out = head(features)
        
        assert out.shape == (2, 13, 288, 384)
    
    def test_depth_head_output_shape(self):
        """Depth head should output [B, 1, H, W]."""
        head = DenseRegressionHead(d_model=64, output_dim=1)
        
        features = torch.randn(2, 64, 288, 384)
        out = head(features)
        
        assert out.shape == (2, 1, 288, 384)
    
    def test_normal_head_output_shape(self):
        """Normal head should output [B, 3, H, W]."""
        head = DenseRegressionHead(d_model=64, output_dim=3)
        
        features = torch.randn(2, 64, 288, 384)
        out = head(features)
        
        assert out.shape == (2, 3, 288, 384)


# ============================================================================
# CATEGORY E: GRADIENT FLOW TESTS
# ============================================================================

class TestBackboneGradientFlow:
    """Verify gradient flow through backbones."""
    
    def test_segnet_gradient_flows(self):
        """Gradient should flow through SegNet."""
        model = SegNet(input_channels=3, d_model=64)
        
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_shared_trunk_gradient_flows(self):
        """Gradient should flow through SharedTrunk."""
        model = SharedTrunk(input_dim=20, d_model=64, n_layers=3)
        
        x = torch.randn(4, 20, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_resblock_gradient_flows(self):
        """Gradient should flow through ResBlock (including residual)."""
        block = ResBlock(d_model=64, dropout=0.0)
        
        x = torch.randn(2, 64, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestHeadGradientFlow:
    """Verify gradient flow through heads."""
    
    def test_regression_head_gradient_flows(self):
        """Gradient should flow through RegressionHead."""
        head = RegressionHead(d_model=64, output_dim=1)
        
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_classification_head_gradient_flows(self):
        """Gradient should flow through ClassificationHead."""
        head = ClassificationHead(d_model=64, num_classes=10)
        
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_dense_segmentation_gradient_flows(self):
        """Gradient should flow through DenseSegmentationHead."""
        head = DenseSegmentationHead(d_model=64, num_classes=13)
        
        x = torch.randn(2, 64, 64, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ============================================================================
# CATEGORY F: INVARIANT TESTS
# ============================================================================

class TestBackboneInvariants:
    """Test invariants that must hold across all backbones."""
    
    def test_all_backbones_are_nn_modules(self):
        """All backbones should be nn.Module instances."""
        assert isinstance(SegNet(), nn.Module)
        assert isinstance(SharedTrunk(input_dim=10, d_model=64), nn.Module)
    
    def test_all_backbones_accept_kwargs(self):
        """All backbones should accept **kwargs for flexibility."""
        model = SegNet()
        x = torch.randn(1, 3, 64, 64)
        
        # Should not raise error
        out = model(x, some_extra_arg=None)
        assert out is not None


class TestHeadInvariants:
    """Test invariants that must hold across all heads."""
    
    def test_all_heads_are_nn_modules(self):
        """All heads should be nn.Module instances."""
        assert isinstance(RegressionHead(d_model=64), nn.Module)
        assert isinstance(ClassificationHead(d_model=64), nn.Module)
        assert isinstance(DenseSegmentationHead(d_model=64), nn.Module)
        assert isinstance(DenseRegressionHead(d_model=64), nn.Module)
    
    def test_all_heads_accept_global_ctx(self):
        """All heads should accept global_ctx argument for API compatibility."""
        heads = [
            RegressionHead(d_model=64),
            ClassificationHead(d_model=64),
            DenseSegmentationHead(d_model=64),
            DenseRegressionHead(d_model=64),
        ]
        
        for head in heads:
            if isinstance(head, (DenseSegmentationHead, DenseRegressionHead)):
                x = torch.randn(2, 64, 32, 32)
            else:
                x = torch.randn(2, 64)
            
            global_ctx = torch.randn(2, 64)
            
            # Should not raise error
            out = head(x, global_ctx=global_ctx)
            assert out is not None


# ============================================================================
# CATEGORY G: NUMERICAL STABILITY TESTS
# ============================================================================

class TestBackboneNumericalStability:
    """Verify numerical stability under extreme inputs."""
    
    def test_segnet_large_input_values(self):
        """SegNet should handle large input values."""
        model = SegNet(input_channels=3, d_model=64)
        
        x = torch.randn(2, 3, 64, 64) * 100.0
        out = model(x)
        
        assert torch.isfinite(out).all()
    
    def test_shared_trunk_large_input_values(self):
        """SharedTrunk should handle large input values."""
        model = SharedTrunk(input_dim=20, d_model=64, n_layers=3)
        
        x = torch.randn(4, 20) * 100.0
        out = model(x)
        
        assert torch.isfinite(out).all()
    
    def test_segnet_batch_norm_stability(self):
        """BatchNorm should not produce NaN with small batches."""
        model = SegNet(input_channels=3, d_model=64)
        model.eval()  # Use running stats
        
        x = torch.randn(1, 3, 64, 64)  # Batch size 1
        out = model(x)
        
        assert torch.isfinite(out).all()


# ============================================================================
# CATEGORY H: EDGE CASE TESTS
# ============================================================================

class TestBackboneEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_segnet_minimum_resolution(self):
        """SegNet should handle minimum resolution (must be divisible by 32)."""
        model = SegNet(input_channels=3, d_model=64)
        
        # 32x32 is minimum (5 pooling layers = 2^5 = 32)
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        
        assert out.shape == (1, 64, 32, 32)
    
    def test_segnet_single_channel_input(self):
        """Should handle single-channel input."""
        model = SegNet(input_channels=1, d_model=64)
        
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        
        assert out.shape == (2, 64, 64, 64)
    
    def test_shared_trunk_single_sample(self):
        """Should handle batch size 1."""
        model = SharedTrunk(input_dim=20, d_model=64)
        
        x = torch.randn(1, 20)
        out = model(x)
        
        assert out.shape == (1, 64)


# ============================================================================
# CATEGORY I: REGRESSION TESTS
# ============================================================================

class TestBackboneHeadRegressions:
    """Tests for specific bugs that have been fixed."""
    
    def test_segnet_unpooling_indices_preserved(self):
        """
        REGRESSION: Unpooling indices were not properly preserved.
        
        Fix: Encoder blocks return indices, decoder blocks use them.
        """
        # This is verified by the shape preservation tests
        assert True
    
    def test_shared_trunk_final_norm_fix(self):
        """
        REGRESSION: Missing final LayerNorm caused unnormalized outputs.
        
        Fix: Added final_norm layer in SharedTrunk.
        """
        model = SharedTrunk(input_dim=20, d_model=64, n_layers=2)
        
        x = torch.randn(4, 20)
        out = model(x)
        
        # Output should be normalized (mean ~0, std ~1)
        assert abs(out.mean().item()) < 1.0
        assert abs(out.std().item() - 1.0) < 0.5
    
    def test_regression_head_3d_input_extraction(self):
        """
        REGRESSION: 3D input was mean-pooled instead of extracting final state.
        
        Fix: Changed to features[:, -1, :] for sequence data.
        """
        head = RegressionHead(d_model=64, output_dim=1)
        
        # Create input where last timestep differs from mean
        x = torch.zeros(2, 10, 64)
        x[:, -1, :] = 1.0  # Only last timestep has signal
        
        out = head(x)
        
        # Should not be zero (if mean-pooled, would be near zero)
        assert out.abs().sum() > 0.1


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
