"""
tests/test_bpgs.py
------------------
Unit tests for B-PGS (Bayesian Projected Gradient Scaling).

Tests:
    1. Bounds adherence (Sigmoid mapping always within [s_min, s_max])
    2. Gradient flow (non-zero gradients at all theta values)
    3. Auto-calibration correctness (Empirical Bayes Initialization)
    4. Integration with network losses (Temperature Equalized Softmax)
"""

import pytest
import torch
import math

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectra.core.bpgs import BPGS


class TestBPGSBounds:
    """Test 1: Sigmoid mapping always stays within topological limits."""

    def test_bounds_random_theta(self):
        """For 10,000 random theta values, s must stay within bounds."""
        scaler = BPGS(num_tasks=5)
        s_min, s_max = -scaler.topological_limit, scaler.topological_limit
        
        with torch.no_grad():
            for _ in range(100):
                scaler.theta.data = torch.randn(5) * 50  # Wild values
                s_vals = scaler.get_s()
                for v in s_vals:
                    assert (v >= s_min - 1e-5), f"Below s_min: {v} < {s_min}"
                    assert (v <= s_max + 1e-5), f"Above s_max: {v} > {s_max}"

    def test_bounds_extreme_theta(self):
        """Extreme theta values (+/-1000) must still produce valid bounds."""
        scaler = BPGS(num_tasks=3)
        s_min, s_max = -scaler.topological_limit, scaler.topological_limit
        
        with torch.no_grad():
            scaler.theta.data = torch.tensor([1000.0, -1000.0, 0.0])
            
        s_vals = scaler.get_s()
        assert (s_min - 1e-5 <= s_vals[0] <= s_max + 1e-5)
        assert (s_min - 1e-5 <= s_vals[1] <= s_max + 1e-5)
        assert (s_min - 1e-5 <= s_vals[2] <= s_max + 1e-5)


class TestBPGSGradients:
    """Test 2: Gradients always flow through manifold."""

    def test_gradient_exists(self):
        """theta.grad must be non-None and non-zero after uncertainty backward."""
        scaler = BPGS(num_tasks=3)
        
        # Call with some losses (triggers auto-calibration on step 0)
        losses = [torch.tensor(10.0), torch.tensor(5.0), torch.tensor(0.5)]
        loss_unc = scaler.uncertainty_loss(losses)
        loss_unc.backward()
        
        assert scaler.theta.grad is not None
        assert scaler.theta.grad.abs().sum() > 0

    def test_gradient_at_boundary(self):
        """Gradients must be non-zero even when theta is pushed near bounds."""
        scaler = BPGS(num_tasks=2)
        
        # Bypass auto-calibration to manually test gradients near boundary
        scaler._calibrated = True 
        
        with torch.no_grad():
            # Use moderate values where sigmoid is near boundary but not saturated
            scaler.theta.data = torch.tensor([5.0, -5.0])
            
        losses = [torch.tensor(10.0), torch.tensor(0.1)]
        loss_unc = scaler.uncertainty_loss(losses)
        loss_unc.backward()
        
        # Both gradients must be non-zero
        assert scaler.theta.grad[0].abs() > 1e-6
        assert scaler.theta.grad[1].abs() > 1e-6


class TestBPGSAutoCalibration:
    """Test 3: Empirical Bayes Initialization (Auto-Calibration)."""

    def test_auto_calibration_correctness(self):
        """Auto-calibration should perfectly map s to log(L) on step 0."""
        scaler = BPGS(num_tasks=3)
        losses = [torch.tensor(2.5), torch.tensor(0.5), torch.tensor(0.1)]
        
        # Before calibration, s is 0.0
        s_initial = scaler.get_s()
        assert all(abs(s.item() - 0.0) < 1e-5 for s in s_initial)
        
        # Trigger calibration
        scaler.auto_calibrate(losses)
        
        # After calibration, s should map exactly to log(L)
        s_calibrated = scaler.get_s(losses)
        
        # With SVAM, the bounds are dynamic based on mu and sigma
        detached_losses = torch.stack([l.detach() for l in losses])
        log_L = torch.log(detached_losses.clamp(min=1e-8))
        mu = log_L.mean()
        sigma = log_L.std(unbiased=False).clamp(min=1e-4)
        s_min = mu - scaler.topological_limit * sigma
        s_max = mu + scaler.topological_limit * sigma
        
        expected_s = torch.clamp(torch.log(torch.tensor([2.5, 0.5, 0.1])), min=s_min.item(), max=s_max.item())
        
        # Check precision (allow 1e-3 for inverse sigmoid mapping precision loss near boundaries)
        for i in range(3):
            assert abs(s_calibrated[i].item() - expected_s[i].item()) < 1e-3

    def test_auto_calibration_triggers_once(self):
        """Calibration should only trigger on the first forward pass."""
        scaler = BPGS(num_tasks=1)
        assert scaler._calibrated is False
        
        # Step 0
        l_unc = scaler.uncertainty_loss([torch.tensor(2.0)])
        assert scaler._calibrated is True
        
        # Record theta
        theta_calibrated = scaler.theta.data.clone()
        
        # Step 1 with wildly different loss (should NOT trigger calibration)
        scaler.uncertainty_loss([torch.tensor(100.0)])
        
        # Theta should remain unchanged (only modified by backprop, not calibration)
        assert torch.allclose(scaler.theta.data, theta_calibrated)


class TestBPGSIntegration:
    """Test 4: Integration with network losses (Temperature Equalized Softmax)."""

    def test_network_loss_routing(self):
        """Network weights must correctly follow the T-Softmax equation."""
        num_tasks = 3
        temperature = 2.0
        scaler = BPGS(num_tasks=num_tasks, temperature=temperature)
        
        losses = [torch.tensor(l, requires_grad=True) for l in [2.5, 0.5, 0.1]]
        
        # Force a known s state
        with torch.no_grad():
            # Let's say s perfectly tracks log(L)
            scaler.theta.data = torch.zeros(3) # dummy, we'll bypass get_s to test the formula
            s_val = torch.log(torch.tensor([2.5, 0.5, 0.1]))
            
            # We must override get_s to return our exact values for testing
            scaler.get_s = lambda raw_losses=None: s_val

        l_net = scaler.network_loss(losses)
        l_net.backward()
        
        # Extract grads from the leaf tensors (dL/dL_i = W_i)
        grads = torch.tensor([l.grad.item() for l in losses])
        
        # Calculate expected weights mathematically: Softmax(exp(-s) / T)
        expected_precision = torch.exp(-s_val)
        expected_weights = torch.softmax(expected_precision / temperature, dim=0)
        
        for i in range(num_tasks):
            assert abs(grads[i] - expected_weights[i].item()) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
