import torch
from spectra.core.bpgs_pure import BPGSPure

def test_bpgs_pure_gradients():
    print("Running B-PGS Pure Gradient Detachment Test...")
    num_tasks = 2
    bpgs = BPGSPure(num_tasks=num_tasks, s_min=-2.0, s_max=2.0, s_init=0.0)
    
    # Fake network parameters
    network_params = torch.tensor([1.0, 2.0], requires_grad=True)
    
    # Fake task losses
    losses = [network_params[0]**2, network_params[1]**2]
    
    # 1. Test Network Loss
    loss_net = bpgs.network_loss(losses)
    loss_net.backward()
    
    print(f"Network Params Grad: {network_params.grad}")
    print(f"BPGS Theta Grad (after net loss): {bpgs.theta.grad}")
    
    # theta.grad should be None or Zero because network_loss uses precision.detach()
    if bpgs.theta.grad is not None and torch.abs(bpgs.theta.grad).sum() > 0:
        print("FAIL: Theta received gradients from network_loss!")
    else:
        print("PASS: Theta isolated from network_loss.")
        
    # 2. Test Uncertainty Loss
    bpgs.theta.grad = None
    network_params.grad = None
    
    # Re-compute losses because graph was consumed
    losses = [network_params[0]**2, network_params[1]**2]
    loss_unc = bpgs.uncertainty_loss(losses)
    loss_unc.backward()
    
    print(f"Network Params Grad (after unc loss): {network_params.grad}")
    print(f"BPGS Theta Grad: {bpgs.theta.grad}")
    
    # network_params.grad should be None or Zero because uncertainty_loss uses loss.detach()
    if network_params.grad is not None and torch.abs(network_params.grad).sum() > 0:
        print("FAIL: Network params received gradients from uncertainty_loss!")
    else:
        print("PASS: Network params isolated from uncertainty_loss.")
        
    if bpgs.theta.grad is None or torch.abs(bpgs.theta.grad).sum() == 0:
        print("FAIL: Theta DID NOT receive gradients from uncertainty_loss!")
    else:
        print("PASS: Theta optimized by uncertainty_loss.")

if __name__ == "__main__":
    test_bpgs_pure_gradients()
