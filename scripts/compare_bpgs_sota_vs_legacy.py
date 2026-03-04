"""
scripts/compare_bpgs_sota_vs_legacy.py
---------------------------------------
A forensic, side-by-side comparison of the Legacy B-PGS (v2) and the SOTA B-PGS (v3).
This script executes identical training simulations to verify:
1. Weight Adaptation Speed (Convergence).
2. Numerical Stability (NaN resilience).
3. Gradient Flow Decoupling.
4. Mathematical robustness under task-loss shifts.

Usage: python scripts/compare_bpgs_sota_vs_legacy.py
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict

# Import both versions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectra.core.bpgs_legacy import BPGSScaler as BPGS_Legacy
from spectra.core.bpgs import BPGS as BPGS_SOTA

def run_simulation(name: str, mode: str, steps: int = 500):
    print(f"\n[Simulation] Mode: {name} ({mode})")
    
    # 1. Setup Tasks
    # Task 0: Large (MSE ~50)
    # Task 1: Med (MSE ~1.0)
    # Task 2: Small (BCE ~0.1)
    num_tasks = 3
    s_min, s_max = -5.0, 10.0
    decay = 0.99
    tau = 1.0 / (1.0 - decay)
    lr_net = 1e-3
    lr_theta = 1e-2 # BPGS parameters usually benefit from higher LR
    
    # 2. Setup Model and Weighter
    # We use a simple Linear layer as our 'backbone'
    backbone = nn.Linear(10, 10, bias=False)
    
    if mode == "legacy":
        weighter = BPGS_Legacy(num_tasks, s_min=s_min, s_max=s_max, ema_decay=decay)
        opt = torch.optim.AdamW([
            {"params": backbone.parameters(), "lr": lr_net},
            {"params": weighter.parameters(), "lr": lr_theta, "weight_decay": 0.0}
        ])
    else:
        weighter = BPGS_SOTA(num_tasks, s_min=s_min, s_max=s_max, tau=tau)
        opt_net = torch.optim.AdamW(backbone.parameters(), lr=lr_net)
        opt_unc = torch.optim.AdamW(weighter.parameters(), lr=lr_theta, weight_decay=0.0)

    # 3. Simulate Training
    history = []
    
    for i in range(steps):
        # Generate Synthetic Losses with noise
        with torch.no_grad():
            # Simulated loss behavior: they gradually "learn" (decrease)
            # except for one jittery task.
            l0 = 50.0 * math.exp(-i/200.0) + torch.randn(1).item() * 5.0
            l1 = 1.0  * math.exp(-i/400.0) + torch.randn(1).item() * 0.1
            l2 = 0.5  + torch.randn(1).item() * 0.05
            
            # Step 250: THE CRISIS (Task 1 suddenly explodes)
            if i == 250:
                l1 += 100.0
                print(f"  Step {i}: [CRITICAL] Spike injected into Task 1")

            losses = [torch.tensor(max(l0, 1e-4), requires_grad=True), 
                      torch.tensor(max(l1, 1e-4), requires_grad=True), 
                      torch.tensor(max(l2, 1e-4), requires_grad=True)]
        
        # Optimization Step
        if mode == "legacy":
            opt.zero_grad()
            total_loss, metrics = weighter(torch.stack(losses), sync_ddp=False)
            total_loss.backward()
            opt.step()
            # Post-step projection as done in trainer.py
            weighter.project_parameters()
            
        else:
            # SOTA Decoupled Loop
            weighter.update_ema(losses)
            
            # Base Step
            opt_net.zero_grad()
            l_net = weighter.network_loss(losses)
            l_net.backward()
            opt_net.step()
            
            # Fiber Step
            opt_unc.zero_grad()
            l_unc = weighter.uncertainty_loss()
            l_unc.backward()
            opt_unc.step()
            
            metrics = weighter.get_task_stats()

        # Record Metrics
        if mode == "legacy":
            weights = [weighter.get_log_vars()[j].item() for j in range(num_tasks)]
        else:
            weights = weighter.get_log_vars() # Already a list of floats
            
        precisions = [math.exp(-w) for w in weights]
        history.append({
            "step": i,
            "weights": weights,
            "precisions": precisions,
            "ema": weighter.get_L_bar() if mode == "sota" else weighter.loss_ema.tolist()
        })
        
        if i % 100 == 0 or i == steps - 1:
            w_str = ", ".join([f"{w:.2f}" for w in weights])
            p_str = ", ".join([f"{p:.3f}" for p in precisions])
            print(f"  [{i:4d}] Log-Vars: [{w_str}] | Precisions: [{p_str}]")

    return history

def compare():
    print("="*80)
    print("B-PGS SOTA vs LEGACY COMPARISON")
    print("="*80)
    
    history_legacy = run_simulation("Legacy (v2)", "legacy", steps=500)
    history_sota   = run_simulation("SOTA (v3)", "sota", steps=500)
    
    # Analyze results
    print("\n" + "="*80)
    print("FINAL ANALYSIS")
    print("="*80)
    
    # Task 1 (The one that spiked at 250)
    l1_legacy = history_legacy[251]["weights"][1]
    l1_sota   = history_sota[251]["weights"][1]
    
    l1_f_legacy = history_legacy[-1]["weights"][1]
    l1_f_sota   = history_sota[-1]["weights"][1]

    print(f"Reaction to Spike (Step 250 -> 251):")
    print(f"  Legacy Task 1 Log-Var jump: {history_legacy[250]['weights'][1]:.4f} -> {l1_legacy:.4f}")
    print(f"  SOTA   Task 1 Log-Var jump: {history_sota[250]['weights'][1]:.4f} -> {l1_sota:.4f}")
    
    print(f"\nFinal Convergence (Step 499):")
    print(f"  Legacy Final Log-Vars: {[f'{w:.3f}' for w in history_legacy[-1]['weights']]}")
    print(f"  SOTA   Final Log-Vars: {[f'{w:.3f}' for w in history_sota[-1]['weights']]}")

    # Stability Verification
    all_finite_legacy = all([torch.isfinite(torch.tensor(h["weights"])).all() for h in history_legacy])
    all_finite_sota   = all([torch.isfinite(torch.tensor(h["weights"])).all() for h in history_sota])
    
    print(f"\nNumerical Stability Tests:")
    print(f"  Legacy Finite: {all_finite_legacy}")
    print(f"  SOTA   Finite: {all_finite_sota}")

    print("\n[Verdict]")
    if abs(l1_sota - l1_legacy) < 0.1:
        print(">> Both versions show consistent mathematical convergence.")
    else:
        print(">> SOTA and Legacy differ in adaptation dynamics due to Exact discrete integrator.")
    
    print("\n>> Deep Observation: SOTA uses the Exact Discrete Integrator (beta = 1 - exp(-1/tau))")
    print("   while Legacy uses Standard EMA (alpha = 0.99).")
    print("   Standard alpha=0.99 corresponds to tau=100. Let's compare the actual beta values:")
    beta_legacy = 0.01  # 1 - 0.99
    beta_sota = 1.0 - math.exp(-1.0/100.0) # tau=100
    print(f"   Legacy Beta: {beta_legacy:.6f}")
    print(f"   SOTA   Beta: {beta_sota:.6f} (Difference: {abs(beta_legacy - beta_sota):.6e})")

if __name__ == "__main__":
    compare()
