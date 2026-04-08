import torch
from spectra.baselines.uwso import UWSOWeighter

def test_uwso_robustness():
    """
    Verify that SOTA UW-SO implementation ensures total loss is always >= 0
    and weights form a valid convex combination.
    
    NOTE: UW-SO uses EMA-smoothed losses for weight computation, not raw batch losses.
    This means the weight for a given task may not immediately correspond to the
    current batch's loss ranking, especially in early training.
    """
    print("Testing UWSO Robustness (SOTA Formulation)...")
    num_tasks = 3
    weighter = UWSOWeighter(num_tasks=num_tasks, temperature=1.0)
    
    # Test cases: [loss_task_0, loss_task_1, loss_task_2]
    test_cases = [
        torch.tensor([100.0, 1.0, 0.1]),   # Diverse scales
        torch.tensor([1e-6, 1e-7, 1e-8]),  # Very small losses (converged)
        torch.tensor([10.0, 10.0, 10.0]),  # Equal losses
        torch.tensor([3000.0, 0.5, 0.01])  # Extreme outlier
    ]
    
    for losses in test_cases:
        total, metrics = weighter(losses, sync_ddp=False)
        
        # 1. Non-negativity check
        assert total.item() >= 0, f"Total loss should be >= 0, got {total.item()} for losses {losses}"
        
        # 2. Boundedness check (Convex combination)
        assert total.item() >= losses.min().item() - 1e-7, "Total loss should be >= min(losses)"
        assert total.item() <= losses.max().item() + 1e-7, "Total loss should be <= max(losses)"
        
        # 3. Weight validity check
        weights = [metrics[f"uwso/weight_{i}"] for i in range(num_tasks)]
        
        # All weights should be non-negative
        for w in weights:
            assert w >= 0, f"Weights should be >= 0, got {weights}"
        
        # Weights should sum to approximately 1 (softmax output)
        weight_sum = sum(weights)
        assert abs(weight_sum - 1.0) < 1e-5, f"Weights should sum to 1, got {weight_sum}"
        
        print(f"Losses: {losses.tolist()}")
        print(f"Weights: {weights}")
        print(f"Total Loss: {total.item():.6f}")
        print("-" * 30)

if __name__ == "__main__":
    try:
        test_uwso_robustness()
        print("\n✅ All UW-SO robustness tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
