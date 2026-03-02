import torch
from spectra.baselines.pcgrad import PCGradWeighter
import torch.nn as nn
from torch.cuda.amp import GradScaler
def sim_pcgrad_amp_underflow():
    """
    Simulates the exact conditions of Epoch 0 -> 1 collapse where PCGrad gradient surgery
    might undergo underflow/overflow due to AMP scaler, leading to broken weights.
    We simulate 2 tasks: BCE (imbalanced) and CrossEntropy (balanced).
    """
    print("--- PCGrad AMP Precision Physics Simulation ---")

    # 1. Setup minimal 2-task surrogate model (simulating the shared trunk output)
    num_tasks = 2
    weighter = PCGradWeighter(num_tasks=num_tasks)
    
    # Param tensor mimicking a linear layer in the shared trunk (d_model=512)
    p1 = nn.Parameter(torch.randn(128, 512, dtype=torch.float32) * 0.01) # Small initial weights
    shared_params = [p1]
    
    # 2. Setup AMP Scaler (simulating PyTorch Lightning's handling)
    scaler = GradScaler()
    
    # Simulated inputs
    inputs = torch.randn(64, 512, dtype=torch.float16).cuda() if torch.cuda.is_available() else torch.randn(64, 512)
    p1.data = p1.data.to(inputs.device)
    weighter = weighter.to(inputs.device)
    
    print(f"Device: {inputs.device}")
    
    # We will simulate 3 steps to see if gradients collapse
    for step in range(3):
        # Forward pass (simulating autocast)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            # Shared representations
            shared_rep = torch.matmul(inputs, p1.T) # (64, 128)
            
            # Task 1: Highly imbalanced BCE (Sepsis Outcome) -> Very small gradients for negative class
            labels_t1 = (torch.rand(64) < 0.03).float().to(inputs.device)
            logits_t1 = shared_rep[:, 0]
            loss_t1 = torch.nn.functional.binary_cross_entropy_with_logits(logits_t1, labels_t1, pos_weight=torch.tensor([3.0]).to(inputs.device))
            
            # Task 2: Standard CE (Phase classification) -> Normal gradients
            labels_t2 = torch.randint(0, 3, (64,)).to(inputs.device)
            logits_t2 = shared_rep[:, 1:4]
            loss_t2 = torch.nn.functional.cross_entropy(logits_t2, labels_t2)
        
        # Backward Pass (Simulating SPECTRAModule.training_step where PCGrad intercepts)
        # Scaled losses
        scaled_loss_t1 = scaler.scale(loss_t1)
        scaled_loss_t2 = scaler.scale(loss_t2)
        
        task_losses = [scaled_loss_t1, scaled_loss_t2]
        
        # We assume backward_and_project is handling unscaling correctly OR letting optimizer do it
        # Let's run PCGrad surgery
        shared_params[0].grad = None # Zero grad
        metrics = weighter.backward_and_project(
            task_losses=task_losses,
            shared_params=shared_params,
            skip_backward=False
        )
        
        # After surgery, gradients are STILL SCALED by the scaler!
        # Print gradient norm (scaled)
        scaled_gn = torch.norm(shared_params[0].grad).item()
        print(f"\nStep {step+1}:")
        print(f"Conflicts Detected: {metrics['pcgrad/total_conflicts']}")
        print(f"Scaled Gradient Norm: {scaled_gn:.6f}")
        
        # Unscale for logging/clipping
        scaler.unscale_(torch.optim.SGD(shared_params, lr=0.1)) # Dummy optimizer just for unscale
        unscaled_gn = torch.norm(shared_params[0].grad).item()
        print(f"UNSCALED Gradient Norm: {unscaled_gn:.6f}")
        
        # SIMULATE THE BUG: If unscaled GN is extremely small (e.g. 1e-4) or NaN, surgery broke it
        if torch.isnan(shared_params[0].grad).any():
            print("🚨 CRITICAL: Gradients became NaN during PCGrad surgery!")
        elif unscaled_gn < 1e-5:
            print("🚨 CRITICAL: Gradient Underflow. PCGrad wiped out the signal.")
            
        scaler.step(torch.optim.SGD(shared_params, lr=0.1))
        scaler.update()

if __name__ == "__main__":
    sim_pcgrad_amp_underflow()
