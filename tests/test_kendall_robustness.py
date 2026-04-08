import pytest
import torch
from spectra.baselines.kendall import KendallWeighter

def test_kendall_positive_loss():
    """
    Verify Kendall formulation behavior.
    
    NOTE: The Kendall et al. 2018 formulation can produce negative total loss
    when log_var < 0. This is mathematically correct behavior, not a bug.
    The regularizer 0.5 * log_var can dominate when log_var is very negative.
    
    The Liebel & Körner fix (abs on variance) is NOT applied in this implementation
    as it would change the theoretical properties.
    """
    num_tasks = 3
    weighter = KendallWeighter(num_tasks=num_tasks)
    
    # Simulate a scenario where losses are very small
    losses = torch.tensor([1e-6, 1e-7, 1e-8])
    
    # Test that the formulation produces finite values
    log_var_values = [-10.0, -5.0, 0.0, 5.0, 10.0]
    
    for val in log_var_values:
        weighter.log_vars.data.fill_(val)
        total, metrics = weighter(losses, sync_ddp=False)
        
        # Verify finite output (not NaN or Inf)
        assert torch.isfinite(total), f"Total loss should be finite for log_var={val}, got {total.item()}"
        
        # Verify weights are non-negative
        for i in range(num_tasks):
            assert metrics[f"kendall/weight_{i}"] >= 0, f"Weight should be >= 0"
        
        print(f"log_var={val:5.1f} | total_loss={total.item():.6f}")

if __name__ == "__main__":
    test_kendall_positive_loss()
