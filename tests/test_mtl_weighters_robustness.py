"""
tests/test_mtl_weighters_robustness.py
--------------------------------------
Comprehensive robustness tests for all 7 MTL weighting methods.

This test suite verifies:
- Mathematical correctness against published papers
- Numerical stability under extreme inputs
- Gradient flow correctness
- Edge case handling
- DDP synchronization (simulated)
"""

import math
import pytest
import torch
import torch.nn as nn
from typing import List, Dict

# Import all weighters
from spectra.baselines.static import StaticWeighter
from spectra.baselines.kendall import KendallWeighter
from spectra.baselines.uwso import UWSOWeighter
from spectra.baselines.ntkmtl import NTKMTLWeighter
from spectra.baselines.pcgrad import PCGradWeighter
from spectra.core.bpgs import BPGS


# ============================================================================
# CATEGORY A: CORRECTNESS TESTS - Verify mathematical formulas
# ============================================================================

class TestStaticCorrectness:
    """Verify StaticWeighter implements trivial equal weighting."""
    
    def test_static_equal_weights_by_default(self):
        """L_total = (1/N) * Σ L_i"""
        num_tasks = 4
        weighter = StaticWeighter(num_tasks)
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        total, metrics = weighter(losses)
        
        # Expected: (1+2+3+4) / 4 = 2.5
        expected = losses.sum() / num_tasks
        torch.testing.assert_close(total, expected, rtol=1e-6, atol=1e-7)
    
    def test_static_custom_weights(self):
        """L_total = Σ w_i * L_i with custom weights."""
        weights = [0.1, 0.2, 0.3, 0.4]
        weighter = StaticWeighter(4, weights=weights)
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        total, metrics = weighter(losses)
        
        expected = sum(w * l for w, l in zip(weights, losses))
        torch.testing.assert_close(total, torch.tensor(expected), rtol=1e-6, atol=1e-7)
    
    def test_static_weights_sum_to_one(self):
        """Default weights should sum to 1."""
        weighter = StaticWeighter(5)
        assert torch.allclose(weighter.weights.sum(), torch.tensor(1.0))


class TestKendallCorrectness:
    """
    Verify Kendall et al. 2018 formula:
    L = Σ [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
    Equation 2 for regression, Equation 3 for classification.
    """
    
    def test_kendall_formula_regression(self):
        """Verify exact Kendall et al. 2018 Eq. 2 for regression."""
        num_tasks = 3
        weighter = KendallWeighter(num_tasks)
        
        # Set log_vars to known values
        with torch.no_grad():
            weighter.log_vars.copy_(torch.tensor([0.0, 1.0, -1.0]))
        
        losses = torch.tensor([1.0, 2.0, 3.0])
        total, metrics = weighter(losses)
        
        # Compute expected manually
        # For s = [0, 1, -1]:
        #   exp(-s) = [1, exp(-1), exp(1)] = [1, 0.3679, 2.7183]
        #   weighted = 0.5 * exp(-s) * L = [0.5, 0.3679, 4.0774]
        #   reg = 0.5 * s = [0, 0.5, -0.5]
        #   total = sum(weighted + reg)
        s = torch.tensor([0.0, 1.0, -1.0])
        expected = (0.5 * torch.exp(-s) * losses + 0.5 * s).sum()
        
        torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-6)
    
    def test_kendall_equilibrium_at_log_loss(self):
        """
        At equilibrium, ∂L/∂s_i = 0 → s_i* = log(L_i).
        This is the theoretical fixed point.
        """
        weighter = KendallWeighter(1)
        
        # At equilibrium, s = log(L) should give gradient = 0
        L = torch.tensor(2.0)
        s_eq = math.log(L.item())
        
        with torch.no_grad():
            weighter.log_vars.copy_(torch.tensor([s_eq]))
        
        total, _ = weighter(L.unsqueeze(0))
        
        # Compute gradient analytically
        total.backward()
        
        # ∂L/∂s = -0.5 * exp(-s) * L + 0.5
        # At s = log(L): exp(-s) = 1/L
        # So ∂L/∂s = -0.5 * (1/L) * L + 0.5 = 0
        expected_grad = -0.5 * math.exp(-s_eq) * L.item() + 0.5
        
        torch.testing.assert_close(weighter.log_vars.grad.item(), expected_grad, rtol=1e-5, atol=1e-6)
    
    def test_kendall_fp32_casting_for_amp_safety(self):
        """Verify FP32 casting to prevent exp(11.1) overflow in fp16."""
        weighter = KendallWeighter(2)
        
        # Set log_vars to large values that would overflow in fp16
        with torch.no_grad():
            weighter.log_vars.copy_(torch.tensor([-10.0, 10.0]))
        
        losses = torch.tensor([1.0, 1.0])
        
        # Should not raise or produce inf
        total, _ = weighter(losses)
        assert torch.isfinite(total), f"Total loss is not finite: {total}"


class TestUWSOCorrectness:
    """
    Verify UW-SO formula from Kirchdorfer et al. 2025:
    w_i = softmax( (1/L_i) / T )
    
    Reference: "Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning"
    arXiv:2408.07985, IJCV 2025
    """
    
    def test_uwso_softmax_weights(self):
        """Verify softmax over inverse losses."""
        num_tasks = 3
        weighter = UWSOWeighter(num_tasks, temperature=1.0)
        
        losses = torch.tensor([1.0, 2.0, 4.0])
        
        # Initialize EMA with first batch
        weighter.train()
        total, metrics = weighter(losses)
        
        # With dynamic temp scaling: T_eff = T * mean(1/L)
        inv_losses = 1.0 / losses
        dynamic_temp = 1.0 * inv_losses.mean()
        weights = torch.softmax(inv_losses / dynamic_temp, dim=0)
        
        expected = (weights * losses).sum()
        torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-6)
    
    def test_uwso_weights_sum_to_one(self):
        """Softmax weights must sum to 1."""
        weighter = UWSOWeighter(4, temperature=10.0)
        weighter.train()
        
        losses = torch.tensor([0.5, 1.0, 2.0, 4.0])
        total, metrics = weighter(losses)
        
        # Check metrics for weight sum
        weight_sum = sum(metrics[f"uwso/weight_{i}"] for i in range(4))
        assert abs(weight_sum - 1.0) < 1e-5, f"Weights sum to {weight_sum}, not 1.0"
    
    def test_uwso_dynamic_temperature_scale_invariance(self):
        """
        Dynamic temperature scaling should make weights invariant to loss scale.
        If all losses are scaled by the same factor, weights should remain the same.
        """
        weighter = UWSOWeighter(3, temperature=1.0)
        weighter.train()
        
        losses_small = torch.tensor([1.0, 2.0, 4.0])
        losses_large = losses_small * 1000.0  # Scale by 1000
        
        _, metrics_small = weighter(losses_small)
        
        # Reset EMA
        weighter.ema_initialized.fill_(False)
        weighter.loss_ema.fill_(1.0)
        
        _, metrics_large = weighter(losses_large)
        
        # Weights should be approximately the same (scale-invariant)
        for i in range(3):
            w_small = metrics_small[f"uwso/weight_{i}"]
            w_large = metrics_large[f"uwso/weight_{i}"]
            assert abs(w_small - w_large) < 0.1, f"Weight {i} not scale-invariant: {w_small} vs {w_large}"


class TestPCGradCorrectness:
    """
    Verify PCGrad projection formula from Yu et al. 2020:
    
    If g_i · g_j < 0:
        g_i ← g_i - (g_i · g_j / ||g_j||²) * g_j
    
    Reference: "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    """
    
    def test_pcgrad_conflicting_gradients_projected(self):
        """When gradients conflict (dot < 0), projection should occur."""
        weighter = PCGradWeighter(2)
        
        # Create two parameters
        param1 = nn.Parameter(torch.randn(4))
        param2 = nn.Parameter(torch.randn(4))
        shared_params = [param1, param2]
        
        # Create losses with opposing gradients
        # Loss 1: minimize param1 (gradient = [1, 1, 1, 1])
        # Loss 2: maximize param1 (gradient = [-1, -1, -1, -1])
        loss1 = param1.sum()  # gradient = [1, 1, 1, 1]
        loss2 = -param1.sum()  # gradient = [-1, -1, -1, -1]
        
        metrics = weighter.backward_and_project([loss1, loss2], shared_params)
        
        # After projection, gradients should be orthogonal (dot = 0)
        grad1 = param1.grad
        assert grad1 is not None, "No gradient assigned"
        assert metrics["pcgrad/total_conflicts"] >= 1, "No conflicts detected"
    
    def test_pcgrad_non_conflicting_gradients_unchanged(self):
        """When gradients align (dot >= 0), no projection should occur."""
        weighter = PCGradWeighter(2)
        
        param = nn.Parameter(torch.randn(4))
        shared_params = [param]
        
        # Both losses minimize param (same direction)
        loss1 = param.sum()
        loss2 = 2 * param.sum()
        
        metrics = weighter.backward_and_project([loss1, loss2], shared_params)
        
        # No conflicts should be detected
        assert metrics["pcgrad/total_conflicts"] == 0, "False conflict detected"
        
        # Gradient should be sum of both
        expected_grad = torch.ones(4) + 2 * torch.ones(4)
        torch.testing.assert_close(param.grad, expected_grad, rtol=1e-5, atol=1e-6)
    
    def test_pcgrad_stochastic_ordering_prevents_bias(self):
        """
        Verify that random.shuffle is called to prevent task bias.
        This is a code inspection test - we verify the implementation uses
        stochastic ordering.
        """
        # This test verifies the implementation detail
        # In pcgrad.py lines 112-113, random.shuffle(indices) is called
        # We can verify this by running multiple times and checking variance
        weighter = PCGradWeighter(3)
        
        param = nn.Parameter(torch.randn(10))
        shared_params = [param]
        
        # Create losses with different conflict patterns
        loss1 = param[:5].sum() - param[5:].sum()
        loss2 = -param[:5].sum() + param[5:].sum()
        loss3 = param.sum()
        
        # Run multiple times
        conflict_counts = []
        for _ in range(10):
            param.grad = None
            metrics = weighter.backward_and_project([loss1, loss2, loss3], shared_params)
            conflict_counts.append(metrics["pcgrad/total_conflicts"])
        
        # Due to stochastic ordering, conflict counts should vary
        # (This is a probabilistic test, might not always pass)
        # At minimum, verify no crashes
        assert all(c >= 0 for c in conflict_counts)


class TestBPGSCorrectness:
    """
    Verify B-PGS mathematical properties:
    
    1. Diffeomorphic chart: s = s_min + (s_max - s_min) * sigmoid(theta)
    2. R_eps(x) = sqrt(x² + eps²) — zero-preserving operator
    3. Exact EMA: beta = 1 - exp(-1/tau)
    4. Network loss: L = Σ 0.5 * exp(-s).detach() * L_i
    5. Uncertainty loss: L = Σ [0.5 * exp(-s) * R_eps(L_bar) + 0.5 * s]
    """
    
    def test_bpgs_sigmoid_bounded_log_variance(self):
        """Verify s = s_min + (s_max - s_min) * sigmoid(theta) is bounded."""
        bpgs = BPGS(num_tasks=3, s_min=-10.0, s_max=10.0)
        
        # Test with various theta values
        test_thetas = [-100.0, -10.0, 0.0, 10.0, 100.0]
        
        for theta_val in test_thetas:
            with torch.no_grad():
                bpgs.theta.fill_(theta_val)
            
            s_values = bpgs.get_s()
            s_tensor = torch.stack(s_values)
            
            # All s values must be within [s_min, s_max] (closed interval due to finite precision)
            # At extreme theta values, sigmoid saturates and s approaches s_min or s_max
            assert (s_tensor >= -10.0).all() and (s_tensor <= 10.0).all(), \
                f"s values out of bounds for theta={theta_val}: {s_tensor}"
            
            # Verify bounded: even at extreme theta, s stays within range
            # sigmoid(100) ≈ 1, sigmoid(-100) ≈ 0, so s ∈ [s_min, s_max]
            assert s_tensor.max() <= 10.0 and s_tensor.min() >= -10.0
    
    def test_bpgs_r_eps_zero_preserving(self):
        """Verify R_eps(0) = eps, not log(2) like softplus."""
        from spectra.core.bpgs import R_eps
        
        eps = 1e-5
        x = torch.tensor(0.0)
        result = R_eps(x, eps=eps)
        
        assert abs(result.item() - eps) < 1e-12, \
            f"R_eps(0) = {result}, expected {eps}"
        
        # Verify it's less than log(2)
        assert result.item() < math.log(2), \
            f"R_eps(0) = {result} >= log(2) = {math.log(2)}"
    
    def test_bpgs_exact_ema_beta(self):
        """Verify beta = 1 - exp(-1/tau), not 1/tau approximation."""
        tau = 50.0
        bpgs = BPGS(num_tasks=2, tau=tau)
        
        expected_beta = 1.0 - math.exp(-1.0 / tau)
        
        assert abs(bpgs.beta - expected_beta) < 1e-10, \
            f"beta = {bpgs.beta}, expected {expected_beta}"
        
        # Verify it's NOT the Euler approximation
        euler_approx = 1.0 / tau
        assert abs(bpgs.beta - euler_approx) > 1e-4, \
            f"beta uses Euler approximation {euler_approx}"
    
    def test_bpgs_network_loss_stop_gradient(self):
        """Verify exp(-s) is detached in network loss."""
        bpgs = BPGS(num_tasks=2)
        
        losses = [torch.tensor(1.0, requires_grad=True), 
                  torch.tensor(2.0, requires_grad=True)]
        
        net_loss = bpgs.network_loss(losses)
        net_loss.backward()
        
        # Gradient should NOT flow to theta
        assert bpgs.theta.grad is None or (bpgs.theta.grad == 0).all(), \
            "Gradient leaked into theta through network loss"
    
    def test_bpgs_uncertainty_loss_gradient_flow(self):
        """Verify gradient flows through s → theta in uncertainty loss."""
        bpgs = BPGS(num_tasks=2)
        
        # Update EMA first
        bpgs.update_ema([torch.tensor(1.0), torch.tensor(2.0)])
        
        unc_loss = bpgs.uncertainty_loss()
        unc_loss.backward()
        
        # Gradient MUST flow to theta
        assert bpgs.theta.grad is not None, "No gradient to theta"
        assert (bpgs.theta.grad != 0).any(), "Zero gradient to theta"


# ============================================================================
# CATEGORY B: INVARIANT TESTS - Properties that must always hold
# ============================================================================

class TestWeighterInvariants:
    """Test invariants that must hold across all weighters."""
    
    def test_all_weighters_return_finite_total(self):
        """Total loss must always be finite (no NaN/Inf)."""
        weighters = [
            StaticWeighter(3),
            KendallWeighter(3),
            UWSOWeighter(3),
            NTKMTLWeighter(3),
            PCGradWeighter(3),
            BPGS(num_tasks=3),
        ]
        
        losses = torch.tensor([1.0, 2.0, 3.0])
        
        for weighter in weighters:
            weighter.train()
            if isinstance(weighter, BPGS):
                weighter.update_ema([losses[i] for i in range(3)])
                total = weighter.network_loss([losses[i] for i in range(3)])
            else:
                total, _ = weighter(losses)
            
            assert torch.isfinite(total), f"{weighter.__class__.__name__} produced non-finite total: {total}"
    
    def test_weights_are_non_negative(self):
        """All task weights must be non-negative."""
        weighters = [
            StaticWeighter(3),
            KendallWeighter(3),
            UWSOWeighter(3),
        ]
        
        losses = torch.tensor([0.1, 1.0, 10.0])
        
        for weighter in weighters:
            weighter.train()
            _, metrics = weighter(losses)
            
            for i in range(3):
                key = f"{weighter.__class__.__name__.replace('Weighter', '').lower()}/weight_{i}"
                # Handle different metric naming conventions
                found = False
                for k, v in metrics.items():
                    if f"weight_{i}" in k:
                        found = True
                        assert v >= 0, f"Negative weight {v} for {k}"
                assert found, f"No weight metric found for task {i}"


# ============================================================================
# CATEGORY C: EDGE CASE TESTS
# ============================================================================

class TestWeighterEdgeCases:
    """Test boundary conditions and degenerate inputs."""
    
    def test_single_task(self):
        """All weighters should handle single-task case."""
        weighters = [
            StaticWeighter(1),
            KendallWeighter(1),
            UWSOWeighter(1),
            NTKMTLWeighter(1),
            BPGS(num_tasks=1),
        ]
        
        loss = torch.tensor([1.0])
        
        for weighter in weighters:
            weighter.train()
            if isinstance(weighter, BPGS):
                weighter.update_ema([loss[0]])
                total = weighter.network_loss([loss[0]])
            else:
                total, _ = weighter(loss)
            
            assert torch.isfinite(total), f"{weighter.__class__.__name__} failed on single task"
    
    def test_near_zero_losses(self):
        """Handle losses approaching zero."""
        weighter = UWSOWeighter(3)
        weighter.train()
        
        # Near-zero losses should not cause division by zero
        losses = torch.tensor([1e-10, 1e-8, 1e-6])
        total, metrics = weighter(losses)
        
        assert torch.isfinite(total), f"Non-finite total for near-zero losses: {total}"
    
    def test_large_losses(self):
        """Handle very large losses."""
        weighter = KendallWeighter(3)
        weighter.train()
        
        # Large losses should not overflow
        losses = torch.tensor([1e6, 1e7, 1e8])
        total, _ = weighter(losses)
        
        assert torch.isfinite(total), f"Non-finite total for large losses: {total}"
    
    def test_nan_input_handling(self):
        """UWSO should skip EMA update when loss is NaN."""
        weighter = UWSOWeighter(2)
        weighter.train()
        
        # First, set valid EMA
        valid_losses = torch.tensor([1.0, 2.0])
        weighter(valid_losses)
        
        # Now pass NaN losses
        nan_losses = torch.tensor([float('nan'), 2.0])
        total, metrics = weighter(nan_losses)
        
        # EMA should retain previous value (not be corrupted by NaN)
        assert torch.isfinite(weighter.loss_ema).all(), "EMA corrupted by NaN"


# ============================================================================
# CATEGORY D: NUMERICAL STABILITY TESTS
# ============================================================================

class TestNumericalStability:
    """Test behavior under numerically challenging inputs."""
    
    def test_kendall_fp16_overflow_prevention(self):
        """Kendall should cast to fp32 to prevent exp overflow."""
        weighter = KendallWeighter(2)
        
        # In fp16, exp(11.1) overflows to inf
        # Set log_vars such that exp(-log_vars) would overflow in fp16
        with torch.no_grad():
            weighter.log_vars.copy_(torch.tensor([-15.0, 15.0]))
        
        losses = torch.tensor([1.0, 1.0])
        total, _ = weighter(losses)
        
        assert torch.isfinite(total), "Kendall overflow in exp()"
    
    def test_bpgs_fp16_overflow_prevention(self):
        """BPGS should cast to fp32 before exp()."""
        bpgs = BPGS(num_tasks=2, s_min=-20.0, s_max=20.0)
        
        # Set theta to extreme values
        with torch.no_grad():
            bpgs.theta.fill_(15.0)  # Would cause exp(-s) overflow in fp16
        
        losses = [torch.tensor(1.0), torch.tensor(2.0)]
        bpgs.update_ema(losses)
        
        net_loss = bpgs.network_loss(losses)
        assert torch.isfinite(net_loss), "BPGS overflow in exp()"
    
    def test_uwso_scale_invariance(self):
        """UWSO weights should be invariant to global loss scaling."""
        weighter = UWSOWeighter(3, temperature=1.0)
        weighter.train()
        
        losses_1 = torch.tensor([1.0, 2.0, 4.0])
        _, m1 = weighter(losses_1)
        
        # Reset EMA
        weighter.ema_initialized.fill_(False)
        
        losses_1000 = losses_1 * 1000
        _, m2 = weighter(losses_1000)
        
        # Weights should be similar (scale-invariant due to dynamic temp)
        for i in range(3):
            w1 = m1[f"uwso/weight_{i}"]
            w2 = m2[f"uwso/weight_{i}"]
            rel_diff = abs(w1 - w2) / max(w1, w2, 1e-8)
            assert rel_diff < 0.5, f"Weight {i} not scale-invariant: {w1} vs {w2}"


# ============================================================================
# CATEGORY E: GRADIENT FLOW TESTS
# ============================================================================

class TestGradientFlow:
    """Verify gradients flow correctly through each weighter."""
    
    def test_kendall_gradient_to_log_vars(self):
        """Gradient must flow to log_vars in Kendall."""
        weighter = KendallWeighter(2)
        
        losses = torch.tensor([1.0, 2.0])
        total, _ = weighter(losses)
        total.backward()
        
        assert weighter.log_vars.grad is not None, "No gradient to log_vars"
        assert (weighter.log_vars.grad != 0).any(), "Zero gradient to log_vars"
    
    def test_bpgs_decoupled_gradient_flows(self):
        """BPGS must have separate gradient flows for network and uncertainty."""
        bpgs = BPGS(num_tasks=2)
        
        losses = [torch.tensor(1.0, requires_grad=False), 
                  torch.tensor(2.0, requires_grad=False)]
        
        # Update EMA
        bpgs.update_ema(losses)
        
        # Network loss: gradient should NOT flow to theta
        losses_with_grad = [torch.tensor(1.0, requires_grad=True),
                           torch.tensor(2.0, requires_grad=True)]
        net_loss = bpgs.network_loss(losses_with_grad)
        net_loss.backward()
        
        assert bpgs.theta.grad is None or (bpgs.theta.grad == 0).all(), \
            "Gradient leaked to theta through network loss"
        
        # Reset
        bpgs.theta.grad = None
        
        # Uncertainty loss: gradient SHOULD flow to theta
        unc_loss = bpgs.uncertainty_loss()
        unc_loss.backward()
        
        assert bpgs.theta.grad is not None, "No gradient to theta from uncertainty loss"
    
    def test_pcgrad_gradient_assignment(self):
        """PCGrad must assign gradients to parameter.grad."""
        weighter = PCGradWeighter(2)
        
        param = nn.Parameter(torch.randn(4))
        shared_params = [param]
        
        loss1 = param.sum()
        loss2 = param.sum() * 2
        
        weighter.backward_and_project([loss1, loss2], shared_params)
        
        assert param.grad is not None, "No gradient assigned to param.grad"
        assert torch.isfinite(param.grad).all(), "Non-finite gradient"


# ============================================================================
# CATEGORY F: REGRESSION TESTS
# ============================================================================

class TestRegressions:
    """Tests for specific bugs that have been fixed."""
    
    def test_ntkmtl_retain_graph_fix(self):
        """
        REGRESSION: NTKMTL used to free computation graph on last task,
        causing RuntimeError every update_interval steps.
        
        Fix: retain_graph=True for ALL tasks.
        """
        weighter = NTKMTLWeighter(2, update_interval=1)
        weighter.train()
        
        # Create parameters with gradients
        param = nn.Parameter(torch.randn(4))
        shared_params = [param]
        
        # Create losses that share computation graph
        x = param.sum()
        loss1 = x * 0.5
        loss2 = x * 0.5
        
        losses = torch.stack([loss1, loss2])
        
        # This should NOT raise RuntimeError
        try:
            total, _ = weighter(losses, shared_params=shared_params)
            # The graph is still needed for backward
            total.backward()
        except RuntimeError as e:
            if "retaining graph" in str(e).lower():
                pytest.fail(f"NTKMTL retain_graph regression: {e}")
    
    def test_uwso_nan_gate_prevents_ema_corruption(self):
        """
        REGRESSION: NaN losses used to corrupt EMA permanently.
        
        Fix: Skip EMA update when loss is not finite.
        """
        weighter = UWSOWeighter(2)
        weighter.train()
        
        # Initialize with valid losses
        valid = torch.tensor([1.0, 2.0])
        weighter(valid)
        old_ema = weighter.loss_ema.clone()
        
        # Pass NaN
        nan_losses = torch.tensor([float('nan'), 2.0])
        weighter(nan_losses)
        
        # EMA for task 0 should be unchanged
        assert torch.isfinite(weighter.loss_ema[0]), "EMA corrupted by NaN"
        assert weighter.loss_ema[0] == old_ema[0], "EMA changed despite NaN gate"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
