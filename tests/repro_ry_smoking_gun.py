import torch
import torch.nn as nn
from spectra.baselines.pcgrad import PCGradWeighter
import random

def test_smoking_gun_1():
    print("--- Running Smoking Gun #1: Per-Parameter vs Full-Vector Projection ---")
    torch.manual_seed(0)
    random.seed(0)
    
    # We create two parameters
    p1 = nn.Parameter(torch.randn(10))
    p2 = nn.Parameter(torch.randn(10))
    shared_params = [p1, p2]
    
    # We construct task gradients such that:
    # 1. Global dot product is POSITIVE (no project should happen)
    # 2. Local dot product for p1 is NEGATIVE (buggy code WILL project)
    
    # Task 0: just random
    g0_p1 = torch.randn(10)
    g0_p2 = torch.randn(10)
    
    # Task 1: 
    # Make p1 part negative relative to g0_p1
    g1_p1 = -0.1 * g0_p1
    # Make p2 part strongly positive relative to g0_p2 so global sum is positive
    g1_p2 = 5.0 * g0_p2
    
    global_dot = torch.dot(torch.cat([g0_p1, g0_p2]), torch.cat([g1_p1, g1_p2]))
    local_dot_p1 = torch.dot(g0_p1, g1_p1)
    
    print(f"Global dot: {global_dot.item():.4f}")
    print(f"Local dot p1: {local_dot_p1.item():.4f}")
    
    assert global_dot > 0, "Global dot must be positive for this test"
    assert local_dot_p1 < 0, "Local dot p1 must be negative for this test"
    
    weighter = PCGradWeighter(num_tasks=2)
    
    # Simulate task_grads for project_and_assign
    task_grads = [
        [g0_p1.clone(), g0_p2.clone()],
        [g1_p1.clone(), g1_p2.clone()]
    ]
    
    # Reset grads
    for p in shared_params:
        p.grad = None
        
    metrics = weighter.project_and_assign(task_grads, shared_params)
    
    # In full-vector PCGrad (paper), there should be NO projection because global_dot > 0
    # In per-parameter PCGrad (current bug), p1 will be projected because local_dot_p1 < 0
    
    # If projection happened, p1.grad will NOT be (g0_p1 + g1_p1)
    expected_p1_grad = g0_p1 + g1_p1
    actual_p1_grad = p1.grad
    
    diff = torch.norm(actual_p1_grad - expected_p1_grad).item()
    print(f"Difference from expected (sum): {diff:.6f}")
    
    if diff > 1e-5:
        print("RESULT: BUG CONFIRMED - p1 was projected despite positive global dot product.")
        print(f"Total conflicts reported: {metrics['pcgrad/total_conflicts']}")
    else:
        print("RESULT: No projection occurred on p1. (This is CORRECT for full-vector PCGrad)")

if __name__ == "__main__":
    test_smoking_gun_1()
