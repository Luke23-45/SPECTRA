import math
import torch
import torch.nn.functional as F
import numpy as np

def R_eps(x, eps=1e-5):
    return torch.sqrt(x.pow(2) + eps**2)

def softplus(x):
    return F.softplus(x)

def check_zero_point_bias():
    print("--- Testing Zero-Point Thermal Bias (R_eps vs Softplus) ---")
    x_zero = torch.tensor(0.0)
    x_tiny = torch.tensor(1e-8)
    x_large = torch.tensor(10.0)
    
    print(f"Softplus(0.0) = {softplus(x_zero).item():.6f} (This is log(2) floor!)")
    print(f"R_eps(0.0)    = {R_eps(x_zero).item():.6f} (Correctly approaches 0)")
    
    print(f"\nSoftplus(1e-8) = {softplus(x_tiny).item():.6f}")
    print(f"R_eps(1e-8)    = {R_eps(x_tiny).item():.6f}")
    
    print(f"\nSoftplus(10.0) = {softplus(x_large).item():.6f}")
    print(f"R_eps(10.0)    = {R_eps(x_large).item():.6f}")
    
    # Check derivative at zero
    x_test = torch.tensor(0.0, requires_grad=True)
    r = R_eps(x_test)
    r.backward()
    print(f"dR_eps/dx at x=0: {x_test.grad.item():.6f}")
    
    x_test2 = torch.tensor(0.0, requires_grad=True)
    s = softplus(x_test2)
    s.backward()
    print(f"dSoftplus/dx at x=0: {x_test2.grad.item():.6f}")

def simulate_uncertainty_landscape(l_bar_val, s_min=-10.0, s_max=10.0):
    print(f"\n--- Uncertainty Landscape (L_bar = {l_bar_val}) ---")
    thetas = torch.linspace(-10, 10, 200)
    s_vals = s_min + (s_max - s_min) * torch.sigmoid(thetas)
    
    # Compute L_unc
    # L_unc = 0.5 * exp(-s) * R_eps(L_bar) + 0.5 * s
    l_bars = torch.full_like(s_vals, l_bar_val)
    r_val = R_eps(l_bars)
    
    weights = torch.exp(-s_vals)
    l_unc = 0.5 * weights * r_val + 0.5 * s_vals
    
    # Find minimum
    min_idx = torch.argmin(l_unc)
    opt_theta = thetas[min_idx].item()
    opt_s = s_vals[min_idx].item()
    opt_weight = weights[min_idx].item()
    
    # Theoretical optimum: exp(-s) * R_eps(l_bar) = 1 => s* = ln(R_eps(l_bar))
    theoretical_s = math.log(R_eps(torch.tensor(l_bar_val)).item())
    
    print(f"Empirical minimum found at theta={opt_theta:.2f}, s={opt_s:.4f}")
    print(f"Theoretical optimum s* = ln(R_eps) = {theoretical_s:.4f}")
    print(f"Resulting precision weight at optimum: {opt_weight:.4f}")
    
    # Check if bounds are active
    if abs(opt_s - s_min) < 0.1:
        print("WARNING: Saturation at s_min bound!")
    if abs(opt_s - s_max) < 0.1:
        print("WARNING: Saturation at s_max bound!")

if __name__ == "__main__":
    check_zero_point_bias()
    simulate_uncertainty_landscape(0.0)      # Perfectly solved task
    simulate_uncertainty_landscape(0.001)    # Almost solved
    simulate_uncertainty_landscape(1.0)      # Standard unit loss
    simulate_uncertainty_landscape(100.0)    # High loss
    simulate_uncertainty_landscape(1e5)      # Extreme loss / Outlier
