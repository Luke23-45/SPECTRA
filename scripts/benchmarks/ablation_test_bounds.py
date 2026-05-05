import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from spectra.core.bpgs import BPGS

def generate_extreme_loss(step, max_steps):
    # Task 0: Huge loss (cross entropy simulation)
    # Task 1: Tiny loss (cosine simulation)
    progress = step / max_steps
    
    ce_mean = 100.0 * np.exp(-3 * progress)
    ce_val = max(0.01, ce_mean + np.random.normal(0, 0.2 * ce_mean))
    
    cos_mean = 0.01 * (1 - 0.9 * progress)
    cos_val = max(0.0001, cos_mean + np.random.normal(0, 0.1 * cos_mean))
    
    return ce_val, cos_val

def run_bounds_ablation(limit_value, steps=1000, lr=0.05):
    print(f"\n--- Testing Topological Limit = {limit_value} ---")
    
    # Initialize BPGS
    temperature = 2.0
    bpgs = BPGS(num_tasks=2, s_mode="stateless", init_mode="fixed")
    
    # Forcefully override the bounds for this ablation test
    s_min = -float(limit_value)
    s_max = float(limit_value)
    bpgs.s_min_v.fill_(s_min)
    bpgs.s_max_v.fill_(s_max)
    
    # Re-initialize theta to center at 0.0 with the new bounds
    bpgs.s_min = s_min
    bpgs.s_max = s_max
    theta_init = bpgs._theta_from_s(0.0)
    bpgs.theta.data.fill_(theta_init)
    
    optimizer = torch.optim.Adam(bpgs.parameters(), lr=lr)
    
    history = {
        's': np.zeros((steps, 2)),
        'raw_precision': np.zeros((steps, 2)),
        'temp_equalized_weight': np.zeros((steps, 2))
    }
    
    for step in range(steps):
        ce_val, cos_val = generate_extreme_loss(step, steps)
        
        l0 = torch.tensor(ce_val, dtype=torch.float32, requires_grad=True)
        l1 = torch.tensor(cos_val, dtype=torch.float32, requires_grad=True)
        raw_losses = [l0, l1]
        
        optimizer.zero_grad()
        
        # Uncertainty updates s
        u_loss = bpgs.uncertainty_loss(raw_losses)
        u_loss.backward()
        optimizer.step()
        
        # Record stats
        with torch.no_grad():
            s = bpgs.get_s()
            raw_precision = torch.exp(-s)
            eq_weights = bpgs.num_tasks * torch.softmax(-s / temperature, dim=0)
            
            history['s'][step, 0] = s[0].item()
            history['s'][step, 1] = s[1].item()
            history['raw_precision'][step, 0] = raw_precision[0].item()
            history['raw_precision'][step, 1] = raw_precision[1].item()
            history['temp_equalized_weight'][step, 0] = eq_weights[0].item()
            history['temp_equalized_weight'][step, 1] = eq_weights[1].item()
            
    # Print Final Status
    print(f"Final s_0 (CE): {history['s'][-1, 0]:.4f} | Final s_1 (Cosine): {history['s'][-1, 1]:.4f}")
    print(f"Raw Precision -> CE: {history['raw_precision'][-1, 0]:.4f} | Cosine: {history['raw_precision'][-1, 1]:.4f}")
    print(f"Temp-Eq Weight-> CE: {history['temp_equalized_weight'][-1, 0]:.4f} | Cosine: {history['temp_equalized_weight'][-1, 1]:.4f}")
    
    return history

def plot_ablation(results, limit_values, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    steps = len(results[limit_values[0]]['s'])
    x = np.arange(steps)
    
    fig, axs = plt.subplots(len(limit_values), 2, figsize=(12, 4 * len(limit_values)))
    
    for i, limit in enumerate(limit_values):
        hist = results[limit]
        
        # Plot Raw Precision
        axs[i, 0].plot(x, hist['raw_precision'][:, 0], label="CE (Huge Loss)")
        axs[i, 0].plot(x, hist['raw_precision'][:, 1], label="Cosine (Tiny Loss)")
        axs[i, 0].set_title(f"Raw Precision (Limit = {limit})")
        axs[i, 0].set_yscale('log')
        axs[i, 0].legend()
        axs[i, 0].grid(True, alpha=0.3)
        
        # Plot Temp Equalized Weight
        axs[i, 1].plot(x, hist['temp_equalized_weight'][:, 0], label="CE (Huge Loss)")
        axs[i, 1].plot(x, hist['temp_equalized_weight'][:, 1], label="Cosine (Tiny Loss)")
        axs[i, 1].set_title(f"Temp-Equalized Weights (Limit = {limit})")
        axs[i, 1].set_ylim(0, 2.5)
        axs[i, 1].legend()
        axs[i, 1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, "bounds_ablation_results.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved ablation plot to {out_path}")

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'stress_test_results')
    
    # We test different limits:
    # 0.5: Extremely tight bounds (little adaptability)
    # 2.0: Our chosen Euler bound (e^2)
    # 5.0: Loose bounds
    # 10.0: Basically unbounded, allowing mathematical explosions
    limit_values = [0.5, 2.0, 5.0, 10.0]
    results = {}
    
    for limit in limit_values:
        results[limit] = run_bounds_ablation(limit)
        
    plot_ablation(results, limit_values, out_dir)

if __name__ == '__main__':
    main()
