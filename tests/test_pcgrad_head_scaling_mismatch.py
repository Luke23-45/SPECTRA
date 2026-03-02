import torch
from spectra.baselines.pcgrad import PCGradWeighter
import torch.nn as nn

def test_pcgrad_head_scaling_mismatch():
    print("--- Testing PCGrad Scaling Mismatch between Backbone and Heads ---\n")
    
    num_tasks = 2
    weighter = PCGradWeighter(num_tasks=num_tasks)
    
    # Simulate a minimal backbone and two heads
    backbone_layer = nn.Linear(10, 10, bias=False)
    head1 = nn.Linear(10, 1, bias=False)
    head2 = nn.Linear(10, 1, bias=False)
    
    # Initialize all weights to 1.0 for easy tracking of gradients
    nn.init.constant_(backbone_layer.weight, 1.0)
    nn.init.constant_(head1.weight, 1.0)
    nn.init.constant_(head2.weight, 1.0)
    shared_params = list(backbone_layer.parameters())
    
    # Forward pass
    x = torch.ones(1, 10)
    shared_features = backbone_layer(x) # Output is 10
    
    out1 = head1(shared_features) # Output is 100
    out2 = head2(shared_features) # Output is 100
    
    # Dummy targets and L2 loss
    target1 = torch.zeros(1, 1)
    target2 = torch.zeros(1, 1)
    
    loss1 = ((out1 - target1) ** 2).mean() # Gradients will be 2 * 100 = 200
    loss2 = ((out2 - target2) ** 2).mean() 
    
    # --- Simulate SPECTRAModule logic ---
    task_losses = [loss1, loss2]
    
    # 1. Autograd for backbone (simulating scaled grads)
    task_grads = []
    for loss in task_losses:
        grads = torch.autograd.grad(loss, shared_params, retain_graph=True)
        task_grads.append(list(grads))
        
    print(f"Task 1 raw backbone grad norm: {torch.norm(task_grads[0][0]):.4f}")
    print(f"Task 2 raw backbone grad norm: {torch.norm(task_grads[1][0]):.4f}")
        
    # 2. PCGrad Surgery (applies mean or sum)
    weighter.project_and_assign(task_grads[:], shared_params)
    backbone_grad_norm_pcgrad = torch.norm(shared_params[0].grad)
    
    # Calculate expected sum
    backbone_grad_norm_sum = torch.norm(task_grads[0][0] + task_grads[1][0])
    
    # 3. Head gradients (computed individually)
    head1_grads = torch.autograd.grad(loss1, head1.parameters(), retain_graph=True)
    head1_grad_norm = torch.norm(head1_grads[0])
    
    print("\n--- Results ---")
    print(f"Backbone gradient norm (PCGrad): {backbone_grad_norm_pcgrad:.4f}")
    print(f"Backbone gradient norm (Expected SUM): {backbone_grad_norm_sum:.4f}")
    print(f"Ratio (PCGrad / Expected SUM): {backbone_grad_norm_pcgrad / backbone_grad_norm_sum:.2f}x")
    
    print("\n🚨 Conclusion:")
    if backbone_grad_norm_pcgrad < backbone_grad_norm_sum * 0.9:
        print("Bug Found! PCGrad is washing out gradients (e.g. using .mean() instead of .sum()), "
              "causing the backbone to learn significantly slower than the task heads.")
    else:
        print("PASS: PCGrad accumulation scale matches the expected multi-task summation scale.")

if __name__ == "__main__":
    test_pcgrad_head_scaling_mismatch()
