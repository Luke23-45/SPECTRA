import pytest
import torch
from spectra.baselines.kendall import KendallWeighter

def test_kendall_positive_loss():
    """
    Verify that Liebel & Körner fix ensures total loss is always >= 0
    even when task losses approach zero.
    """
    num_tasks = 3
    weighter = KendallWeighter(num_tasks=num_tasks)
    
    # Simulate a scenario where losses are very small (risking negative total in vanilla)
    losses = torch.tensor([1e-6, 1e-7, 1e-8])
    
    # Try different log_var values, especially large negative ones that would 
    # make 0.5 * log_var very negative in the original version.
    log_var_values = [-10.0, -5.0, 0.0, 5.0, 10.0]
    
    for val in log_var_values:
        weighter.log_vars.data.fill_(val)
        total, _ = weighter(losses, sync_ddp=False)
        
        assert total.item() >= 0, f"Total loss should be >= 0 for log_var={val}, got {total.item()}"
        print(f"log_var={val:5.1f} | total_loss={total.item():.6f}")

if __name__ == "__main__":
    test_kendall_positive_loss()
