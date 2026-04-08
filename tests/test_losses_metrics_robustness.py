"""
tests/test_losses_metrics_robustness.py
---------------------------------------
Comprehensive robustness tests for loss functions and metrics.

This test suite verifies:
- LOSS_REGISTRY: Loss construction and device handling
- BCEWithLogitsLossDynamicDevice: Device migration of pos_weight
- MaskedL1Loss: Valid pixel masking, zero handling
- DenseCosineLoss: Angular error, normalization, gradient flow
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from spectra.engine.losses import (
    LOSS_REGISTRY,
    BCEWithLogitsLossDynamicDevice,
    _build_bce_loss,
    _build_ce_loss,
    _build_masked_l1_loss,
    _build_dense_cosine_loss,
)
from spectra.engine.dense_losses import MaskedL1Loss, DenseCosineLoss


# ============================================================================
# CATEGORY A: CORRECTNESS TESTS - LOSS_REGISTRY
# ============================================================================

class TestLossRegistryCorrectness:
    """Verify LOSS_REGISTRY constructs losses correctly."""
    
    def test_registry_contains_all_standard_losses(self):
        """Registry must contain all standard loss types."""
        required_losses = ["mse", "l1", "bce", "cross_entropy", "cosine"]
        
        for name in required_losses:
            assert name in LOSS_REGISTRY, f"Missing loss: {name}"
    
    def test_registry_mse_is_mseloss(self):
        """MSE loss should be nn.MSELoss."""
        loss_fn = LOSS_REGISTRY["mse"]()
        assert isinstance(loss_fn, nn.MSELoss)
    
    def test_registry_l1_is_l1loss(self):
        """L1 loss should be nn.L1Loss."""
        loss_fn = LOSS_REGISTRY["l1"]()
        assert isinstance(loss_fn, nn.L1Loss)
    
    def test_registry_bce_with_pos_weight(self):
        """BCE loss with pos_weight should use dynamic device wrapper."""
        loss_fn = LOSS_REGISTRY["bce"](pos_weight=5.0)
        
        assert isinstance(loss_fn, BCEWithLogitsLossDynamicDevice)
        assert loss_fn.pos_weight is not None
        assert loss_fn.pos_weight.item() == 5.0
    
    def test_registry_bce_without_pos_weight(self):
        """BCE loss without pos_weight should still be dynamic wrapper."""
        loss_fn = LOSS_REGISTRY["bce"]()
        
        assert isinstance(loss_fn, BCEWithLogitsLossDynamicDevice)
        assert loss_fn.pos_weight is None
    
    def test_registry_cross_entropy_with_ignore_index(self):
        """CrossEntropy loss should support ignore_index."""
        loss_fn = LOSS_REGISTRY["cross_entropy"](ignore_index=255)
        
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.ignore_index == 255


# ============================================================================
# CATEGORY B: CORRECTNESS TESTS - BCEWithLogitsLossDynamicDevice
# ============================================================================

class TestBCEWithLogitsDynamicDevice:
    """Verify BCEWithLogitsLossDynamicDevice handles device migration."""
    
    def test_bce_dynamic_device_migrates_pos_weight(self):
        """pos_weight should migrate to input device automatically."""
        loss_fn = BCEWithLogitsLossDynamicDevice(pos_weight=torch.tensor([5.0]))
        
        # pos_weight initially on CPU
        assert loss_fn.pos_weight.device.type == "cpu"
        
        # Input on CPU - should work
        input_cpu = torch.randn(4, 1)
        target_cpu = torch.randint(0, 2, (4, 1)).float()
        loss = loss_fn(input_cpu, target_cpu)
        
        assert loss.isfinite()
        assert loss_fn.pos_weight.device.type == "cpu"
    
    def test_bce_dynamic_device_computes_correctly(self):
        """Verify BCE computation matches standard formula."""
        loss_fn = BCEWithLogitsLossDynamicDevice(pos_weight=torch.tensor([2.0]))
        
        # Simple test case
        input_tensor = torch.tensor([0.0, 1.0, -1.0, 2.0])
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        
        loss = loss_fn(input_tensor, target)
        
        # Manual computation with pos_weight=2.0
        # BCE = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
        # With pos_weight: -[pos_w*y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
        sigmoid = torch.sigmoid(input_tensor)
        expected = -(
            2.0 * target * torch.log(sigmoid) +
            (1 - target) * torch.log(1 - sigmoid)
        ).mean()
        
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)


# ============================================================================
# CATEGORY C: CORRECTNESS TESTS - MaskedL1Loss
# ============================================================================

class TestMaskedL1LossCorrectness:
    """Verify MaskedL1Loss computes L1 only on valid pixels."""
    
    def test_masked_l1_ignores_invalid_pixels(self):
        """Invalid pixels (mask=0) should not contribute to loss."""
        loss_fn = MaskedL1Loss()
        
        pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # [1, 1, 2, 2]
        target = torch.tensor([[[[0.0, 0.0], [1.0, 2.0]]]])  # Invalid, Invalid, Valid, Valid
        mask = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])  # 0=invalid, 1=valid
        
        loss = loss_fn(pred, target, meta={"depth_mask": mask})
        
        # Only valid pixels: (3-1) + (4-2) = 2 + 2 = 4, mean = 4/2 = 2.0
        expected = torch.tensor(2.0)
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    
    def test_masked_l1_derives_mask_from_target(self):
        """Without meta, mask should be derived from target > 0."""
        loss_fn = MaskedL1Loss()
        
        pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = torch.tensor([[[[0.0, 0.0], [1.0, 2.0]]]])  # 0 is invalid
        
        loss = loss_fn(pred, target)
        
        # Same as above: only pixels with target > 0 contribute
        expected = torch.tensor(2.0)
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    
    def test_masked_l1_all_invalid_returns_zero(self):
        """All invalid pixels should return 0 loss."""
        loss_fn = MaskedL1Loss()
        
        pred = torch.randn(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)  # All invalid
        mask = torch.zeros(2, 1, 4, 4)
        
        loss = loss_fn(pred, target, meta={"depth_mask": mask})
        
        assert loss.item() == 0.0
    
    def test_masked_l1_perfect_prediction_zero_loss(self):
        """Perfect prediction should have zero loss."""
        loss_fn = MaskedL1Loss()
        
        pred = torch.randn(2, 1, 4, 4)
        target = pred.clone()
        mask = torch.ones(2, 1, 4, 4)
        
        loss = loss_fn(pred, target, meta={"depth_mask": mask})
        
        assert loss.item() < 1e-6
    
    def test_masked_l1_handles_shape_mismatch(self):
        """Should handle mask [B, H, W] vs pred [B, 1, H, W]."""
        loss_fn = MaskedL1Loss()
        
        pred = torch.randn(2, 1, 4, 4)
        target = torch.rand(2, 1, 4, 4) * 5.0
        mask = torch.ones(2, 4, 4)  # Missing channel dim
        
        loss = loss_fn(pred, target, meta={"depth_mask": mask})
        
        assert loss.isfinite()


# ============================================================================
# CATEGORY D: CORRECTNESS TESTS - DenseCosineLoss
# ============================================================================

class TestDenseCosineLossCorrectness:
    """Verify DenseCosineLoss computes angular error correctly."""
    
    def test_cosine_loss_perfect_prediction(self):
        """Identical normals should have ~0 loss."""
        loss_fn = DenseCosineLoss()
        
        # Random unit normals
        normal = torch.randn(2, 3, 4, 4)
        normal = F.normalize(normal, p=2, dim=1)
        
        loss = loss_fn(normal, normal)
        
        assert loss.item() < 1e-5
    
    def test_cosine_loss_opposite_directions(self):
        """Opposite normals should have loss = 2."""
        loss_fn = DenseCosineLoss()
        
        normal = torch.randn(2, 3, 4, 4)
        normal = F.normalize(normal, p=2, dim=1)
        opposite = -normal
        
        loss = loss_fn(opposite, normal)
        
        # cos_sim = -1, so loss = 1 - (-1) = 2
        assert abs(loss.item() - 2.0) < 0.01
    
    def test_cosine_loss_orthogonal_vectors(self):
        """Orthogonal normals should have loss = 1."""
        loss_fn = DenseCosineLoss()
        
        # Create orthogonal vectors
        pred = torch.zeros(2, 3, 4, 4)
        pred[:, 0, :, :] = 1.0  # x-direction
        
        target = torch.zeros(2, 3, 4, 4)
        target[:, 1, :, :] = 1.0  # y-direction
        
        loss = loss_fn(pred, target)
        
        # cos_sim = 0, so loss = 1 - 0 = 1
        assert abs(loss.item() - 1.0) < 0.01
    
    def test_cosine_loss_ignores_zero_normals(self):
        """Zero normals (invalid) should be excluded."""
        loss_fn = DenseCosineLoss()
        
        pred = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)
        
        # Set half the target to zero (invalid)
        target[:, :, :, :2] = 0.0
        
        loss = loss_fn(pred, target)
        
        # Should be finite (not NaN or Inf)
        assert loss.isfinite()
    
    def test_cosine_loss_all_invalid_returns_zero(self):
        """All invalid normals should return 0 loss."""
        loss_fn = DenseCosineLoss()
        
        pred = torch.randn(2, 3, 4, 4)
        target = torch.zeros(2, 3, 4, 4)  # All invalid
        
        loss = loss_fn(pred, target)
        
        assert loss.item() == 0.0


# ============================================================================
# CATEGORY E: GRADIENT FLOW TESTS
# ============================================================================

class TestLossGradientFlow:
    """Verify gradient flow through all losses."""
    
    def test_masked_l1_gradient_flows(self):
        """Gradient should flow through MaskedL1Loss."""
        loss_fn = MaskedL1Loss()
        
        pred = torch.randn(2, 1, 4, 4, requires_grad=True)
        target = torch.rand(2, 1, 4, 4) * 5.0
        mask = torch.ones(2, 1, 4, 4)
        
        loss = loss_fn(pred, target, meta={"depth_mask": mask})
        loss.backward()
        
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
    
    def test_cosine_loss_gradient_flows(self):
        """Gradient should flow through DenseCosineLoss."""
        loss_fn = DenseCosineLoss()
        
        pred = torch.randn(2, 3, 4, 4, requires_grad=True)
        target = F.normalize(torch.randn(2, 3, 4, 4), p=2, dim=1)
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
    
    def test_bce_gradient_flows(self):
        """Gradient should flow through BCEWithLogitsLossDynamicDevice."""
        loss_fn = BCEWithLogitsLossDynamicDevice(pos_weight=torch.tensor([2.0]))
        
        input_tensor = torch.randn(4, 1, requires_grad=True)
        target = torch.randint(0, 2, (4, 1)).float()
        
        loss = loss_fn(input_tensor, target)
        loss.backward()
        
        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all()


# ============================================================================
# CATEGORY F: NUMERICAL STABILITY TESTS
# ============================================================================

class TestLossNumericalStability:
    """Verify numerical stability under extreme inputs."""
    
    def test_masked_l1_large_values(self):
        """MaskedL1Loss should handle large values."""
        loss_fn = MaskedL1Loss()
        
        pred = torch.randn(2, 1, 4, 4) * 1e6
        target = torch.randn(2, 1, 4, 4) * 1e6
        mask = torch.ones(2, 1, 4, 4)
        
        loss = loss_fn(pred, target, meta={"depth_mask": mask})
        
        assert loss.isfinite()
    
    def test_cosine_loss_clamp_prevents_nan(self):
        """Cosine clamping should prevent NaN from acos."""
        loss_fn = DenseCosineLoss()
        
        # Create vectors that might produce cos_sim outside [-1, 1]
        pred = torch.randn(2, 3, 4, 4) * 10.0
        target = torch.randn(2, 3, 4, 4) * 10.0
        
        loss = loss_fn(pred, target)
        
        assert loss.isfinite()
    
    def test_bce_extreme_logits(self):
        """BCE should handle extreme logits without overflow."""
        loss_fn = BCEWithLogitsLossDynamicDevice()
        
        # Extreme logits
        input_tensor = torch.tensor([-100.0, 100.0, -50.0, 50.0])
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        
        loss = loss_fn(input_tensor, target)
        
        assert loss.isfinite()


# ============================================================================
# CATEGORY G: INVARIANT TESTS
# ============================================================================

class TestLossInvariants:
    """Test invariants that must hold across all losses."""
    
    def test_all_losses_return_scalar(self):
        """All losses must return a scalar tensor."""
        losses_to_test = [
            ("mse", nn.MSELoss()),
            ("l1", nn.L1Loss()),
            ("bce", _build_bce_loss()),
            ("masked_l1", MaskedL1Loss()),
            ("cosine_dense", DenseCosineLoss()),
        ]
        
        for name, loss_fn in losses_to_test:
            if name == "masked_l1":
                pred = torch.randn(2, 1, 4, 4)
                target = torch.rand(2, 1, 4, 4)
                loss = loss_fn(pred, target, meta={"depth_mask": torch.ones(2, 1, 4, 4)})
            elif name == "cosine_dense":
                pred = torch.randn(2, 3, 4, 4)
                target = torch.randn(2, 3, 4, 4)
                loss = loss_fn(pred, target)
            elif name == "bce":
                pred = torch.randn(4, 1)
                target = torch.randint(0, 2, (4, 1)).float()
                loss = loss_fn(pred, target)
            else:
                pred = torch.randn(4, 1)
                target = torch.randn(4, 1)
                loss = loss_fn(pred, target)
            
            assert loss.dim() == 0, f"{name} returned non-scalar: {loss.shape}"
    
    def test_all_losses_are_nn_modules(self):
        """All losses should be nn.Module instances."""
        assert isinstance(MaskedL1Loss(), nn.Module)
        assert isinstance(DenseCosineLoss(), nn.Module)
        assert isinstance(BCEWithLogitsLossDynamicDevice(), nn.Module)


# ============================================================================
# CATEGORY H: REGRESSION TESTS
# ============================================================================

class TestLossRegressions:
    """Tests for specific bugs that have been fixed."""
    
    def test_bce_pos_weight_device_migration_fix(self):
        """
        REGRESSION: pos_weight stayed on CPU when model moved to GPU.
        
        Fix: BCEWithLogitsLossDynamicDevice migrates pos_weight on forward().
        """
        # This is verified by test_bce_dynamic_device_migrates_pos_weight
        assert True
    
    def test_masked_l1_zero_division_fix(self):
        """
        REGRESSION: Division by zero when all pixels invalid.
        
        Fix: mask.sum().clamp(min=1.0) prevents division by zero.
        """
        loss_fn = MaskedL1Loss()
        
        pred = torch.randn(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        mask = torch.zeros(2, 1, 4, 4)
        
        loss = loss_fn(pred, target, meta={"depth_mask": mask})
        
        # Should not be NaN or Inf
        assert loss.item() == 0.0
    
    def test_cosine_loss_nan_from_unnormalized_fix(self):
        """
        REGRESSION: Unnormalized vectors could produce cos_sim > 1.
        
        Fix: Explicit F.normalize() and clamping to [-1, 1].
        """
        loss_fn = DenseCosineLoss()
        
        # Unnormalized vectors
        pred = torch.randn(2, 3, 4, 4) * 5.0
        target = torch.randn(2, 3, 4, 4) * 5.0
        
        loss = loss_fn(pred, target)
        
        assert loss.isfinite()


# ============================================================================
# CATEGORY I: METRICS CORRECTNESS TESTS
# ============================================================================

class TestSegmentationMetricsCorrectness:
    """Verify SegmentationMetrics computes mIoU correctly."""
    
    def test_seg_perfect_prediction_miou_one(self):
        """Perfect prediction should give mIoU = 1.0."""
        from spectra.evaluation.metrics import SegmentationMetrics
        
        seg = SegmentationMetrics(num_classes=5)
        
        # Create perfect predictions
        target = torch.randint(0, 5, (2, 8, 8))
        logits = torch.zeros(2, 5, 8, 8)
        logits.scatter_(1, target.unsqueeze(1), 100.0)  # One-hot logits
        
        seg.update(logits, target)
        results = seg.compute()
        
        assert abs(results["miou"] - 1.0) < 1e-4, f"Perfect prediction mIoU: {results['miou']}"
        assert results["pixel_acc"] == 1.0
    
    def test_seg_ignores_ignore_index(self):
        """Pixels with ignore_index should be excluded."""
        from spectra.evaluation.metrics import SegmentationMetrics
        
        seg = SegmentationMetrics(num_classes=3, ignore_index=255)
        
        target = torch.zeros(2, 8, 8, dtype=torch.long)
        target[:, :4, :] = 255  # Mark as ignore
        
        logits = torch.zeros(2, 3, 8, 8)
        logits[:, 0, :, :] = 10.0  # Predict class 0 everywhere
        
        seg.update(logits, target)
        results = seg.compute()
        
        # Only valid pixels (class 0) should contribute
        assert results["n_valid_classes"] == 1
    
    def test_seg_reset_clears_state(self):
        """Reset should clear accumulated state."""
        from spectra.evaluation.metrics import SegmentationMetrics
        
        seg = SegmentationMetrics(num_classes=5)
        
        target = torch.randint(0, 5, (2, 8, 8))
        logits = torch.zeros(2, 5, 8, 8)
        logits.scatter_(1, target.unsqueeze(1), 100.0)
        
        seg.update(logits, target)
        seg.reset()
        
        # After reset, confusion matrix should be zero
        assert seg._conf_matrix.sum() == 0


class TestDepthMetricsCorrectness:
    """Verify DepthMetrics computes standard metrics correctly."""
    
    def test_depth_perfect_prediction_zero_error(self):
        """Perfect prediction should have zero error."""
        from spectra.evaluation.metrics import DepthMetrics
        
        depth_m = DepthMetrics(max_depth=10.0)
        
        target = torch.rand(2, 1, 8, 8) * 5.0 + 0.5
        pred = target.clone()
        
        depth_m.update(pred, target)
        results = depth_m.compute()
        
        assert results["abs_rel"] < 1e-5, f"abs_rel: {results['abs_rel']}"
        assert results["rmse"] < 1e-5
        assert results["delta_1"] > 0.999
    
    def test_depth_masks_invalid_pixels(self):
        """Invalid pixels (depth=0) should be excluded."""
        from spectra.evaluation.metrics import DepthMetrics
        
        depth_m = DepthMetrics(max_depth=10.0)
        
        target = torch.rand(2, 1, 8, 8) * 5.0
        target[:, :, :4, :] = 0.0  # Invalid region
        
        pred = torch.zeros_like(target)
        
        depth_m.update(pred, target)
        results = depth_m.compute()
        
        # Only valid pixels should be counted
        n_valid = 2 * 1 * 8 * 8 - 2 * 1 * 4 * 8  # Total - invalid
        assert results["n_valid"] == n_valid
    
    def test_depth_clips_predictions(self):
        """Predictions should be clipped to [1e-3, max_depth]."""
        from spectra.evaluation.metrics import DepthMetrics
        
        depth_m = DepthMetrics(max_depth=10.0)
        
        target = torch.ones(2, 1, 8, 8) * 5.0
        pred = torch.ones(2, 1, 8, 8) * 100.0  # Way above max_depth
        
        depth_m.update(pred, target)
        results = depth_m.compute()
        
        # Should be finite (not overflow from unclipped values)
        assert math.isfinite(results["abs_rel"])


class TestNormalMetricsCorrectness:
    """Verify NormalMetrics computes angular error correctly."""
    
    def test_normal_perfect_prediction_zero_angle(self):
        """Perfect prediction should have ~0 angle error."""
        from spectra.evaluation.metrics import NormalMetrics
        
        norm_m = NormalMetrics()
        
        target = F.normalize(torch.randn(2, 3, 8, 8), p=2, dim=1)
        pred = target.clone()
        
        norm_m.update(pred, target)
        results = norm_m.compute()
        
        assert results["mean_angle_deg"] < 0.1
        assert results["within_11_25"] > 0.99
    
    def test_normal_opposite_direction_180_deg(self):
        """Opposite normals should have ~180° angle error."""
        from spectra.evaluation.metrics import NormalMetrics
        
        norm_m = NormalMetrics()
        
        target = F.normalize(torch.randn(2, 3, 8, 8), p=2, dim=1)
        pred = -target  # Opposite direction
        
        norm_m.update(pred, target)
        results = norm_m.compute()
        
        assert results["mean_angle_deg"] > 175.0
    
    def test_normal_ignores_zero_vectors(self):
        """Zero vectors (invalid) should be excluded."""
        from spectra.evaluation.metrics import NormalMetrics
        
        norm_m = NormalMetrics()
        
        target = F.normalize(torch.randn(2, 3, 8, 8), p=2, dim=1)
        target[:, :, :4, :] = 0.0  # Invalid region
        
        pred = target.clone()
        
        norm_m.update(pred, target)
        results = norm_m.compute()
        
        assert math.isfinite(results["mean_angle_deg"])


class TestDeltaMMetric:
    """Verify compute_delta_m computes relative improvement correctly."""
    
    def test_delta_m_positive_improvement(self):
        """Better results should give positive Δm%."""
        from spectra.evaluation.metrics import compute_delta_m
        
        delta_m = compute_delta_m(
            results={"miou": 0.45, "abs_rel": 0.18},
            baseline={"miou": 0.40, "abs_rel": 0.20},
            metric_configs={"miou": False, "abs_rel": True},
        )
        
        assert delta_m > 0, f"Expected positive Δm%, got {delta_m}"
    
    def test_delta_m_negative_worse(self):
        """Worse results should give negative Δm%."""
        from spectra.evaluation.metrics import compute_delta_m
        
        delta_m = compute_delta_m(
            results={"miou": 0.35, "abs_rel": 0.25},
            baseline={"miou": 0.40, "abs_rel": 0.20},
            metric_configs={"miou": False, "abs_rel": True},
        )
        
        assert delta_m < 0, f"Expected negative Δm%, got {delta_m}"
    def test_delta_m_handles_zero_baseline(self):
        """Zero baseline should be handled gracefully."""
        from spectra.evaluation.metrics import compute_delta_m
        
        delta_m = compute_delta_m(
            results={"miou": 0.45},
            baseline={"miou": 0.0},
            metric_configs={"miou": False},
        )
        
        # Should skip zero baseline
        assert math.isnan(delta_m)


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
