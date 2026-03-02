"""
tests/test_bpgs_positive_loss.py
---------------------------------
Verification tests for the Decoupled Kendall Formulation fix.

These tests prove that the negative loss divergence is resolved:
1. Total loss is always positive for any realistic inputs
2. Theta/sigma gradient streams are properly decoupled
3. project_parameters() clamps theta correctly
4. Multi-step training simulation stays positive throughout
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectra.core.bpgs import BPGSScaler


class TestPositiveLoss:
    """The total_loss must ALWAYS be positive for non-negative raw losses."""

    def test_positive_loss_typical(self):
        """Typical clinical losses (BCE ~0.5, CE ~1.0) produce positive total."""
        scaler = BPGSScaler(num_tasks=2, s_min=-5.0, s_max=10.0)
        losses = torch.tensor([0.5, 1.0], requires_grad=True)
        total, _ = scaler(losses, sync_ddp=False)
        assert total.item() > 0, f"Total loss should be positive, got {total.item()}"

    def test_positive_loss_large_scale(self):
        """MSE-scale losses (~3000) produce positive total."""
        scaler = BPGSScaler(num_tasks=3, s_min=-5.0, s_max=10.0)
        losses = torch.tensor([3000.0, 2.0, 0.5], requires_grad=True)
        total, _ = scaler(losses, sync_ddp=False)
        assert total.item() > 0, f"Total loss should be positive, got {total.item()}"

    def test_positive_loss_very_small(self):
        """Near-zero losses produce positive total (softplus floor protects)."""
        scaler = BPGSScaler(num_tasks=2, s_min=-5.0, s_max=10.0)
        losses = torch.tensor([0.001, 0.001], requires_grad=True)
        total, _ = scaler(losses, sync_ddp=False)
        assert total.item() > 0, f"Total loss should be positive, got {total.item()}"

    def test_positive_loss_100_random_inputs(self):
        """Over 100 random loss vectors, total loss is ALWAYS positive."""
        scaler = BPGSScaler(num_tasks=4, s_min=-5.0, s_max=10.0)
        torch.manual_seed(42)

        for i in range(100):
            # Random losses in [0.001, 100.0]
            raw = torch.rand(4) * 100 + 0.001
            losses = raw.requires_grad_(True)
            total, _ = scaler(losses, sync_ddp=False)
            assert total.item() > 0, (
                f"Iteration {i}: total_loss={total.item():.6f} is not positive! "
                f"Losses={raw.tolist()}"
            )


class TestDecoupledGradients:
    """Verify that theta_loss detaches precision (network doesn't see σ path)."""

    def test_theta_grad_independent_of_log_vars(self):
        """
        The gradient of total_loss w.r.t. the raw losses should be
        proportional to precision.detach() — i.e., changing theta should
        NOT change the direction of dL/dL_i for the network.
        """
        scaler = BPGSScaler(num_tasks=2, s_min=-2.0, s_max=10.0, use_autocal=False)
        with torch.no_grad():
            scaler.is_calibrated.fill_(True)

        losses = torch.tensor([5.0, 1.0], requires_grad=True)
        total, _ = scaler(losses, sync_ddp=False)
        total.backward()

        # The gradient w.r.t. losses should be 0.5 * precision.detach()
        # (from theta_loss only — sigma_loss uses EMA which is detached from losses)
        log_vars = scaler.get_log_vars()
        expected_grad = 0.5 * torch.exp(-log_vars).detach()
        actual_grad = losses.grad

        for i in range(2):
            diff = abs(actual_grad[i].item() - expected_grad[i].item())
            assert diff < 0.01, (
                f"Task {i}: loss gradient {actual_grad[i]:.4f} != "
                f"expected {expected_grad[i]:.4f} (diff={diff:.6f}). "
                f"Precision may not be properly detached."
            )

    def test_sigma_uses_ema_not_raw_loss(self):
        """
        The sigma_loss should use softplus(loss_ema), NOT the raw batch loss.
        Verify by checking that theta.grad exists and is non-zero even when
        raw losses are zero (because EMA should still have a previous value).
        """
        scaler = BPGSScaler(num_tasks=2, s_min=-2.0, s_max=10.0, use_autocal=False)
        with torch.no_grad():
            scaler.is_calibrated.fill_(True)
            # Set EMA to some positive value
            scaler.loss_ema.copy_(torch.tensor([1.0, 1.0]))

        # Feed zero losses — theta should STILL get a gradient from sigma_loss
        # because sigma_loss uses softplus(EMA) not raw losses
        losses = torch.tensor([0.0, 0.0], requires_grad=True)
        total, _ = scaler(losses, sync_ddp=False)
        total.backward()

        assert scaler.theta.grad is not None, "theta.grad should not be None"
        assert scaler.theta.grad.abs().sum() > 0, (
            "theta.grad should be non-zero (sigma_loss uses EMA, not raw zero losses)"
        )


class TestProjectParameters:
    """Verify project_parameters() correctly clamps theta."""

    def test_clamp_extreme_positive(self):
        scaler = BPGSScaler(num_tasks=2)
        with torch.no_grad():
            scaler.theta.data = torch.tensor([100.0, 50.0])
        scaler.project_parameters()
        assert scaler.theta.data.max() <= 10.0, f"theta not clamped: {scaler.theta.data}"

    def test_clamp_extreme_negative(self):
        scaler = BPGSScaler(num_tasks=2)
        with torch.no_grad():
            scaler.theta.data = torch.tensor([-100.0, -50.0])
        scaler.project_parameters()
        assert scaler.theta.data.min() >= -10.0, f"theta not clamped: {scaler.theta.data}"

    def test_normal_theta_unchanged(self):
        """Theta within [-10, 10] should NOT be modified."""
        scaler = BPGSScaler(num_tasks=3)
        original = torch.tensor([2.0, -3.0, 5.0])
        with torch.no_grad():
            scaler.theta.data = original.clone()
        scaler.project_parameters()
        assert torch.allclose(scaler.theta.data, original), (
            f"theta was modified when it shouldn't be: {scaler.theta.data}"
        )


class TestMultiStepTraining:
    """Simulate multi-step training to verify no negative loss over time."""

    def test_200_step_training_always_positive(self):
        """
        KEY TEST: Simulate 200 optimizer steps — total_loss must NEVER go negative.
        This directly tests the fix for the negative loss divergence from logs_bpgs.md.
        """
        scaler = BPGSScaler(num_tasks=2, s_min=-5.0, s_max=10.0)
        optimizer = torch.optim.Adam(scaler.parameters(), lr=0.01)

        min_loss = float("inf")
        for step in range(200):
            # Simulate varying losses (clinical scenario)
            losses = torch.tensor([
                0.5 + 0.3 * torch.randn(1).item(),
                1.0 + 0.5 * torch.randn(1).item()
            ]).clamp(min=0.01).requires_grad_(True)

            total, _ = scaler(losses, sync_ddp=False)
            min_loss = min(min_loss, total.item())

            assert total.item() > 0, (
                f"Step {step}: total_loss={total.item():.6f} went NEGATIVE! "
                f"This is the exact bug we fixed."
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scaler.project_parameters()

        print(f"✓ 200-step training completed. Min loss: {min_loss:.4f} (always positive)")

    def test_extreme_scale_gap_stays_positive(self):
        """
        Test with 1000x loss scale gap (MSE=3000, BCE=0.5) for 100 steps.
        This is the scenario from the original logs_bpgs.md config.
        """
        scaler = BPGSScaler(num_tasks=2, s_min=-5.0, s_max=10.0)
        optimizer = torch.optim.Adam(scaler.parameters(), lr=0.01)

        for step in range(100):
            losses = torch.tensor([3000.0, 0.5]).requires_grad_(True)
            total, _ = scaler(losses, sync_ddp=False)

            assert total.item() > 0, (
                f"Step {step}: total_loss={total.item():.6f} went NEGATIVE "
                f"with 6000x scale gap!"
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scaler.project_parameters()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
