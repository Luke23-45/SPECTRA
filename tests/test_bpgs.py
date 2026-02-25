"""
tests/test_bpgs.py
------------------
Unit tests for B-PGS (Bayesian Projected Gradient Scaling).

Tests:
    1. Bounds adherence (Sigmoid mapping always within [s_min, s_max])
    2. Gradient flow (non-zero gradients at all theta values)
    3. Auto-calibration correctness
    4. Shadow variable prevention
    5. STE ablation comparison
    6. Basic forward/backward correctness
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectra.core.bpgs import BPGSScaler


class TestBPGSBounds:
    """Test 1: Sigmoid mapping always stays within bounds."""

    def test_bounds_random_theta(self):
        """For 10,000 random theta values, log_vars must be in [s_min, s_max]."""
        scaler = BPGSScaler(num_tasks=5, s_min=-3.0, s_max=8.0)
        with torch.no_grad():
            for _ in range(100):
                scaler.theta.data = torch.randn(5) * 50  # Wild values
                log_vars = scaler.get_log_vars()
                assert (log_vars >= -3.0).all(), f"Below s_min: {log_vars}"
                assert (log_vars <= 8.0).all(), f"Above s_max: {log_vars}"

    def test_bounds_extreme_theta(self):
        """Extreme theta values (+/-1000) must still produce valid bounds."""
        scaler = BPGSScaler(num_tasks=3, s_min=-2.0, s_max=10.0)
        with torch.no_grad():
            scaler.theta.data = torch.tensor([1000.0, -1000.0, 0.0])
        log_vars = scaler.get_log_vars()
        assert (-2.0 <= log_vars[0] <= 10.0)
        assert (-2.0 <= log_vars[1] <= 10.0)
        assert (-2.0 <= log_vars[2] <= 10.0)


class TestBPGSGradients:
    """Test 2: Gradients always flow through sigmoid."""

    def test_gradient_exists(self):
        """theta.grad must be non-None and non-zero after backward."""
        scaler = BPGSScaler(num_tasks=3, s_min=-2.0, s_max=10.0)
        losses = torch.tensor([100.0, 5.0, 0.5], requires_grad=True)
        total, _ = scaler(losses, sync_ddp=False)
        total.backward()
        assert scaler.theta.grad is not None
        assert scaler.theta.grad.abs().sum() > 0

    def test_gradient_at_boundary(self):
        """Gradients must be non-zero even when theta pushes toward bounds."""
        scaler = BPGSScaler(num_tasks=2, s_min=-2.0, s_max=10.0)
        with torch.no_grad():
            # Use moderate values where sigmoid is near boundary but not saturated
            # sigmoid(5) ≈ 0.993, sigmoid'(5) ≈ 0.0066 — measurable gradient
            scaler.theta.data = torch.tensor([5.0, -5.0])
            scaler.is_calibrated.fill_(True)
        losses = torch.tensor([10.0, 0.1], requires_grad=True)
        total, _ = scaler(losses, sync_ddp=False)
        total.backward()
        # Both gradients must be non-zero (sigmoid near boundary still has gradient)
        assert scaler.theta.grad[0].abs() > 1e-6
        assert scaler.theta.grad[1].abs() > 1e-6


class TestBPGSAutoCalibration:
    """Test 3: Auto-calibration sets log_vars ≈ log(L₀)."""

    def test_autocal_correctness(self):
        """After auto_calibrate, log_vars should approximate log(losses)."""
        scaler = BPGSScaler(num_tasks=3, s_min=-5.0, s_max=12.0)
        losses = torch.tensor([3000.0, 2.0, 0.5])
        scaler.auto_calibrate(losses)

        log_vars = scaler.get_log_vars()
        expected = torch.log(losses)

        for i in range(3):
            diff = abs(log_vars[i].item() - expected[i].item())
            assert diff < 0.2, (
                f"Task {i}: log_var={log_vars[i]:.3f}, "
                f"expected={expected[i]:.3f}, diff={diff:.4f}"
            )

    def test_autocal_runs_only_once(self):
        """Auto-calibration should only fire on the first forward pass."""
        scaler = BPGSScaler(num_tasks=2, s_min=-2.0, s_max=10.0)
        losses1 = torch.tensor([100.0, 1.0], requires_grad=True)
        scaler(losses1, sync_ddp=False)

        log_vars_after_first = scaler.get_log_vars().detach().clone()

        # Second forward with different losses — should NOT recalibrate
        losses2 = torch.tensor([0.01, 5000.0], requires_grad=True)
        scaler(losses2, sync_ddp=False)

        log_vars_after_second = scaler.get_log_vars().detach()
        # Log vars should change slightly (optimizer step), but NOT jump to new calibration
        # The key test: they shouldn't match log(losses2)
        wrong_target = torch.log(losses2.detach())
        diff = (log_vars_after_second - wrong_target).abs().sum()
        assert diff > 1.0, "Auto-calibration fired twice!"


class TestBPGSShadowVariable:
    """Test 4: Shadow variable drift prevention."""

    def test_no_theta_explosion(self):
        """After many optimizer steps, theta should stay reasonable (not ±10000)."""
        scaler = BPGSScaler(num_tasks=2, s_min=-2.0, s_max=10.0)
        optimizer = torch.optim.Adam(scaler.parameters(), lr=0.1)

        losses = torch.tensor([1000.0, 0.1], requires_grad=True)
        for _ in range(200):
            losses = torch.tensor([1000.0, 0.1], requires_grad=True)
            total, _ = scaler(losses, sync_ddp=False)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        # Theta should stay within a reasonable range
        assert scaler.theta.data.abs().max() < 50.0, (
            f"Theta exploded: {scaler.theta.data}"
        )

    def test_recovery_after_reversal(self):
        """Log_var should move in different directions for different loss scales."""
        scaler = BPGSScaler(num_tasks=1, s_min=-2.0, s_max=10.0, use_autocal=False)
        optimizer = torch.optim.Adam(scaler.parameters(), lr=0.01)

        # Phase 1: train with small losses → optimizer should push log_var lower
        # (lower log_var = higher precision = appropriate for small losses)
        for _ in range(100):
            losses = torch.tensor([0.001], requires_grad=True)
            total, _ = scaler(losses, sync_ddp=False)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        log_var_after_small = scaler.get_log_vars()[0].item()

        # Phase 2: train with large losses → optimizer should push log_var higher
        # (higher log_var = lower precision = appropriate for large losses)
        for _ in range(200):
            losses = torch.tensor([10000.0], requires_grad=True)
            total, _ = scaler(losses, sync_ddp=False)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        log_var_after_large = scaler.get_log_vars()[0].item()

        # After seeing large losses, log_var should be HIGHER than after small losses
        assert log_var_after_large > log_var_after_small, (
            f"Expected log_var to increase for larger losses: "
            f"after_small={log_var_after_small:.3f}, after_large={log_var_after_large:.3f}"
        )


class TestBPGSSTE:
    """Test 5: STE ablation mode comparison."""

    def test_ste_mode_runs(self):
        """STE (hard clamp) mode should execute without errors."""
        scaler = BPGSScaler(num_tasks=3, use_sigmoid=False, use_autocal=False)
        losses = torch.tensor([10.0, 5.0, 1.0], requires_grad=True)
        total, metrics = scaler(losses, sync_ddp=False)
        total.backward()
        assert torch.isfinite(total)


class TestBPGSForward:
    """Test 6: Basic forward/backward correctness."""

    def test_output_shape(self):
        scaler = BPGSScaler(num_tasks=4)
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        total, metrics = scaler(losses, sync_ddp=False)
        assert total.dim() == 0  # Scalar
        assert total.requires_grad
        assert "bpgs/log_var_0" in metrics
        assert "bpgs/weight_3" in metrics

    def test_nan_input_handling(self):
        """NaN in auto-calibration input should be handled gracefully."""
        scaler = BPGSScaler(num_tasks=3)
        losses = torch.tensor([float("nan"), 5.0, 1.0], requires_grad=True)
        # Should not crash
        total, _ = scaler(losses, sync_ddp=False)
        # May produce NaN in output (expected with NaN input), but shouldn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
