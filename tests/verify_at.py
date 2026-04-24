import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import os

# Ensure we can import spectra
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectra.core.bpgs import BPGS

def test_at_convergence():
    print("="*60)
    print("VERIFYING B-PGS AT (ANALYTICAL TRACKING) CONVERGENCE")
    print("="*60)
    
    num_tasks = 2
    # Set targets: Task 0 (Low Loss), Task 1 (High Loss)
    # Target s0 = log(0.1) approx -2.3
    # Target s1 = log(10.0) approx 2.3
    losses = [torch.tensor(0.1), torch.tensor(10.0)]
    
    # Initialize BPGS with AT-friendly bounds
    bpgs = BPGS(num_tasks=num_tasks, s_min=-5.0, s_max=5.0, s_init=0.0)
    
    # We use a relatively high LR for uncertainty to show fast adaptation
    optimizer = optim.SGD(bpgs.parameters(), lr=0.5)
    
    print(f"Initial s: {bpgs.get_s().detach().numpy()}")
    
    # Run optimization
    for i in range(100):
        optimizer.zero_grad()
        # We simulate the uncertainty_loss call
        loss_unc = bpgs.uncertainty_loss(losses)
        loss_unc.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            print(f"Step {i+1:3d} | s: {bpgs.get_s().detach().numpy()} | loss_unc: {loss_unc.item():.6f}")

    final_s = bpgs.get_s().detach()
    expected_s = torch.tensor([math.log(0.1 + 1e-6), math.log(10.0 + 1e-6)])
    
    print("-" * 30)
    print(f"Final s:    {final_s.numpy()}")
    print(f"Expected s: {expected_s.numpy()}")
    
    # Assertions
    assert torch.allclose(final_s, expected_s, atol=1e-2), "Convergence failed!"
    print("\nPASSED: Convergence Verified")

def test_gradient_stability():
    print("\n" + "="*60)
    print("VERIFYING GRADIENT STABILITY FOR EXTREME LOSSES")
    print("="*60)
    
    bpgs = BPGS(num_tasks=1, s_min=-10.0, s_max=10.0, s_init=0.0)
    
    # Extreme loss: 1 million
    extreme_loss = [torch.tensor(1e6)]
    
    bpgs.theta.grad = None
    l_unc = bpgs.uncertainty_loss(extreme_loss)
    l_unc.backward()
    
    grad_at = bpgs.theta.grad.item()
    print(f"Loss: 1,000,000 | log(L) approx 13.8")
    print(f"QLM Uncertainty Gradient: {grad_at:.4f}")
    
    # The gradient should be finite and manageable (proportional to log(L) * sigmoid_deriv)
    # s = 0, target = 13.8, (s-target) = -13.8.
    # sigmoid_deriv at theta=0 is 0.25. range is 20.
    # grad approx -13.8 * 20 * 0.25 = -69.0
    assert abs(grad_at) < 100.0, f"Gradient too large: {grad_at}"
    print("PASSED: Gradient Stability Verified")

if __name__ == "__main__":
    try:
        test_at_convergence()
        test_gradient_stability()
        print("\n" + "="*60)
        print("ALL B-PGS AT TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        # Note: This will fail until the implementation is updated
        sys.exit(1)
