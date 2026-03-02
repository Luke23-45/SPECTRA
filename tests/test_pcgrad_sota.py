import torch
import torch.nn as nn
from spectra.baselines.pcgrad import PCGradWeighter
import random

def test_conflict_washout_fix():
    """
    Test that Tensor-wise PCGrad catches local conflicts that Global PCGrad would miss.
    """
    num_tasks = 2
    weighter = PCGradWeighter(num_tasks=num_tasks)
    
    # Define two parameter tensors
    # Param 1: Early layer (aligned)
    # Param 2: Late layer (conflicting)
    p1 = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    p2 = nn.Parameter(torch.tensor([1.0]))
    shared_params = [p1, p2]
    
    # Task losses that will produce these gradients
    # Note: We simulate gradients directly for simplicity
    class DummyLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p1, p2, g1, g2):
            ctx.save_for_backward(g1, g2)
            return p1.sum() + p2.sum()
        @staticmethod
        def backward(ctx, grad_output):
            g1, g2 = ctx.saved_tensors
            return g1, g2, None, None

    # Task 1: Positive gradients on both
    g1_p1 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    g1_p2 = torch.tensor([1.0])
    loss1 = DummyLoss.apply(p1, p2, g1_p1, g1_p2)
    
    # Task 2: Weak positive on p1, STRONG negative on p2
    # Global dot product: (5 * 0.1) + (1 * -1.0) = 0.5 - 1.0 = -0.5 (Global would catch this)
    # Let's make Global miss it:
    # Global dot: (5 * 1.0) + (1 * -1.0) = 5.0 - 1.0 = +4.0 (Global misses conflict!)
    g2_p1 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    g2_p2 = torch.tensor([-1.0])
    loss2 = DummyLoss.apply(p1, p2, g2_p1, g2_p2)
    
    task_losses = [loss1, loss2]
    
    # Run surgery
    metrics = weighter.backward_and_project(task_losses, shared_params, skip_backward=True)
    
    # Verification 1: Conflict should be detected in p2
    # C should be > 0
    assert metrics['pcgrad/total_conflicts'] > 0, "Conflict in p2 was missed!"
    
    # Verification 2: p2.grad should be projected
    # Task 1 grad on p2: 1.0
    # Task 2 grad on p2: -1.0
    # Conflict! gi = gi - (gi.gj/|gj|^2) * gj 
    # gi = 1.0 - (1.0 * -1.0 / 1.0) * -1.0 = 1.0 - 1.0 = 0
    # Average of projected gradients for p2 should be 0 (since -1.0 projected onto 1.0 is also 0)
    assert abs(p2.grad.item()) < 1e-6, f"p2.grad should be 0.0, got {p2.grad.item()}"
    
    # Verification 3: p1.grad should be simple average (no conflict)
    # (1.0 + 1.0) / 2 = 1.0
    assert abs(p1.grad[0].item() - 1.0) < 1e-6, "p1.grad should be average of aligned tasks"

    print("SUCCESS: Tensor-wise PCGrad detected local conflict that global alignment would have masked.")

if __name__ == "__main__":
    test_conflict_washout_fix()
