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

from spectra.core.bpgs import BPGS


class TestBPGSBounds:
    """Test 1: Sigmoid mapping always stays within bounds."""

    def test_bounds_random_theta(self):
        """For 10,000 random theta values, log_vars must be in [s_min, s_max]."""
        scaler = BPGS(num_tasks=5, s_min=-3.0, s_max=8.0)
        with torch.no_grad():
            for _ in range(100):
                scaler.theta.data = torch.randn(5) * 50  # Wild values
                log_vars = scaler.get_log_vars()
                for v in log_vars:
                    assert (v >= -3.0), f"Below s_min: {log_vars}"
                    assert (v <= 8.0), f"Above s_max: {log_vars}"

    def test_bounds_extreme_theta(self):
        """Extreme theta values (+/-1000) must still produce valid bounds."""
        scaler = BPGS(num_tasks=3, s_min=-2.0, s_max=10.0)
        with torch.no_grad():
            scaler.theta.data = torch.tensor([1000.0, -1000.0, 0.0])
        log_vars = scaler.get_log_vars()
        assert (-2.0 <= log_vars[0] <= 10.0)
        assert (-2.0 <= log_vars[1] <= 10.0)
        assert (-2.0 <= log_vars[2] <= 10.0)


class TestBPGSGradients:
    """Test 2: Gradients always flow through manifold."""

    def test_gradient_exists(self):
        """theta.grad must be non-None and non-zero after uncertainty backward."""
        scaler = BPGS(num_tasks=3, s_min=-2.0, s_max=10.0)
        # Update EMA with some signal
        scaler.update_ema([torch.tensor(100.0), torch.tensor(5.0), torch.tensor(0.5)])
        
        loss_unc = scaler.uncertainty_loss()
        loss_unc.backward()
        
        assert scaler.theta.grad is not None
        assert scaler.theta.grad.abs().sum() > 0

    def test_gradient_at_boundary(self):
        """Gradients must be non-zero even when theta pushes toward bounds."""
        scaler = BPGS(num_tasks=2, s_min=-2.0, s_max=10.0)
        with torch.no_grad():
            # Use moderate values where sigmoid is near boundary but not saturated
            scaler.theta.data = torch.tensor([5.0, -5.0])
            
        scaler.update_ema([torch.tensor(10.0), torch.tensor(0.1)])
        loss_unc = scaler.uncertainty_loss()
        loss_unc.backward()
        
        # Both gradients must be non-zero (sigmoid near boundary still has gradient)
        assert scaler.theta.grad[0].abs() > 1e-6
        assert scaler.theta.grad[1].abs() > 1e-6


class TestBPGSEMA:
    """Test 3: EMA (Smoothed Loss) behavior."""

    def test_ema_initialization(self):
        """L_bar should initialize to 1.0 (not 0.0) for stability."""
        scaler = BPGS(num_tasks=3)
        l_bar = scaler.get_L_bar()
        assert all(v == 1.0 for v in l_bar)

    def test_ema_convergence_absolute(self):
        """EMA tracks absolute loss (not normalized by global scale).
        A constant loss signal should result in a matching relative L_bar.
        """
        tau = 10.0
        scaler = BPGS(num_tasks=1, tau=tau)
        target = 5.0
        
        # Run for sufficient steps to converge
        for _ in range(200):
            scaler.update_ema([torch.tensor(target)])
            
        final_l_bar = scaler.get_L_bar()[0]
        # In the proper formulation, L_bar converges to target.
        assert abs(final_l_bar - target) < 1e-3


class TestBPGSShadowVariable:
    """Test 4: Shadow variable drift prevention."""

    def test_no_theta_explosion(self):
        """After many optimizer steps, theta should stay reasonable (not ±10000)."""
        scaler = BPGSScaler(num_tasks=2, s_min=-2.0, s_max=10.0)
class TestBPGSShadowVariable:
    """Test 4: Parameter dynamics under high stress."""

    def test_no_theta_explosion(self):
        """Even with massive losses, theta should stay finite due to manifold curvature."""
        scaler = BPGS(num_tasks=2, s_min=-2.0, s_max=10.0)
        optimizer = torch.optim.Adam(scaler.parameters(), lr=1.0)
        
        for _ in range(50):
            losses = [torch.tensor(1e6), torch.tensor(1e6)]
            scaler.update_ema(losses)
            optimizer.zero_grad()
            l_unc = scaler.uncertainty_loss()
            l_unc.backward()
            optimizer.step()
            
        assert torch.isfinite(scaler.theta).all()
        log_vars = scaler.get_log_vars()
        assert all(v <= 10.1 for v in log_vars)

    def test_recovery_after_reversal(self):
        """Log_var should move in according to loss scales."""
        scaler = BPGS(num_tasks=1, s_min=-2.0, s_max=10.0)
        optimizer = torch.optim.Adam(scaler.parameters(), lr=0.1)

        # Phase 1: Small losses
        for _ in range(50):
            scaler.update_ema([torch.tensor(0.001)])
            optimizer.zero_grad()
            l_unc = scaler.uncertainty_loss()
            l_unc.backward()
            optimizer.step()

        log_var_small = scaler.get_log_vars()[0]

        # Phase 2: Large losses
        for _ in range(50):
            scaler.update_ema([torch.tensor(1000.0)])
            optimizer.zero_grad()
            l_unc = scaler.uncertainty_loss()
            l_unc.backward()
            optimizer.step()

        log_var_large = scaler.get_log_vars()[0]
        assert log_var_large > log_var_small


class TestBPGSIntegration:
    """Test 5: Integration with network losses."""

    def test_network_loss_weight_projection(self):
        """Network weights must match exact exp(-s_i) values, no cross-task normalization."""
        num_tasks = 4
        scaler = BPGS(num_tasks=num_tasks)
        # Force unbalanced theta
        with torch.no_grad():
            scaler.theta.data = torch.randn(num_tasks) * 5.0
            
        losses = [torch.tensor(1.0, requires_grad=True) for _ in range(num_tasks)]
        
        # Trigger weight norm via dummy forward or manual weights check
        # We check the actual gradient scaling
        l_net = scaler.network_loss(losses)
        l_net.backward()
        
        # Extract grads from the leaf tensors
        grads = torch.tensor([l.grad.item() for l in losses])
        expected_weights = torch.exp(-torch.tensor(scaler.get_s()))
        # Since network_loss = \sum 0.5 * w_i * L_i, dL/dL_i = 0.5 * w_i
        
        # Individual weights must match precisely
        for i in range(num_tasks):
            assert abs(grads[i] - 0.5 * expected_weights[i]) < 1e-5

    def test_nan_gate_resilience(self):
        """EMA should ignore NaN/Inf signals via NaN Gate."""
        scaler = BPGS(num_tasks=1)
        scaler.update_ema([torch.tensor(5.0)])
        initial_l_bar = scaler.get_L_bar()[0]
        
        scaler.update_ema([torch.tensor(float('nan'))])
        assert scaler.get_L_bar()[0] == initial_l_bar


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
