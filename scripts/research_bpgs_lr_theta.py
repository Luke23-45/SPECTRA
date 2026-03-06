"""
BPGS Theta Learning Rate & Scheduler Sensitivity Study
========================================================
Tests the impact of different uncertainty optimizer configurations on
BPGS convergence quality.

Key question: Should theta use the same LR and cosine schedule as the network?
Hypothesis: The uncertainty parameters need SLOWER adaptation to avoid
premature precision lock-in during late training.
"""
import torch
import torch.nn as nn
import math
import numpy as np

torch.manual_seed(42)

class FullBPGSSim:
    """Full BPGS simulation with configurable theta LR and scheduler."""
    def __init__(self, num_tasks=7, tau=50.0, s_min=-10.0, s_max=10.0):
        self.theta = torch.zeros(num_tasks, requires_grad=True)
        self.s_min = s_min
        self.s_max = s_max
        self.tau = tau
        self.beta = 1 - math.exp(-1/tau)
        self.L_bar = torch.ones(num_tasks)
        self.num_tasks = num_tasks
    
    def get_s(self):
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.theta)
    
    def update_ema(self, losses):
        with torch.no_grad():
            for i, l in enumerate(losses):
                if torch.isfinite(l):
                    self.L_bar[i] = self.L_bar[i] + self.beta * (l.item() - self.L_bar[i])
    
    def network_loss(self, raw_losses):
        s = self.get_s()
        total = 0
        for i, l in enumerate(raw_losses):
            w = torch.exp(-s[i]).detach()
            total = total + 0.5 * w * l
        return total
    
    def uncertainty_loss(self):
        s = self.get_s()
        total = 0
        for i in range(self.num_tasks):
            w = torch.exp(-s[i])
            r = torch.sqrt(self.L_bar[i]**2 + 1e-10)
            total = total + 0.5 * w * r + 0.5 * s[i]
        return total

def cosine_lr_lambda(step, total_steps, warmup_steps, min_ratio):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

def run_experiment(lr_net, lr_theta, use_cosine_for_theta, total_steps=2000, warmup=100, label=""):
    """Simulate BPGS training with configurable theta optimizer settings."""
    torch.manual_seed(42)
    
    # 7 regression tasks at different difficulty levels
    task_complexities = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    num_tasks = len(task_complexities)
    
    bpgs = FullBPGSSim(num_tasks=num_tasks)
    
    # Simple linear models for each task
    models = [torch.tensor([0.5], requires_grad=True) for _ in range(num_tasks)]
    
    opt_net = torch.optim.AdamW(models, lr=lr_net, weight_decay=0.0)
    opt_unc = torch.optim.AdamW([bpgs.theta], lr=lr_theta, weight_decay=0.0)
    
    min_ratio = 1e-6 / lr_net
    min_ratio_t = 1e-6 / lr_theta
    
    history = []
    
    for step in range(total_steps):
        # Cosine schedule for network
        lr_mult = cosine_lr_lambda(step, total_steps, warmup, min_ratio)
        for pg in opt_net.param_groups:
            pg['lr'] = lr_net * lr_mult
        
        # Schedule for theta
        if use_cosine_for_theta:
            lr_mult_t = cosine_lr_lambda(step, total_steps, warmup, min_ratio_t)
        else:
            lr_mult_t = 1.0  # Constant LR for theta
        for pg in opt_unc.param_groups:
            pg['lr'] = lr_theta * lr_mult_t
        
        # Generate batch losses based on task complexities
        raw_losses = []
        for i, complexity in enumerate(task_complexities):
            # Loss = complexity * |model_i - target|^2 + noise
            target = torch.randn(1) * complexity
            pred = models[i] * complexity
            loss = ((pred - target) ** 2).mean()
            raw_losses.append(loss)
        
        # BPGS update cycle
        bpgs.update_ema(raw_losses)
        
        opt_net.zero_grad()
        net_loss = bpgs.network_loss(raw_losses)
        net_loss.backward()
        opt_net.step()
        
        opt_unc.zero_grad()
        unc_loss = bpgs.uncertainty_loss()
        unc_loss.backward()
        opt_unc.step()
        
        if step % 200 == 0 or step == total_steps - 1:
            s_vals = bpgs.get_s().detach()
            weights = [math.exp(-s_vals[i].item()) for i in range(num_tasks)]
            l_bars = bpgs.L_bar.tolist()
            history.append({
                'step': step,
                'net_loss': net_loss.item(),
                'weights': weights,
                'l_bars': l_bars,
                's_vals': s_vals.tolist(),
                'lr_theta_effective': opt_unc.param_groups[0]['lr'],
            })
    
    return history

if __name__ == "__main__":
    configs = [
        {"lr_net": 1e-3, "lr_theta": 1e-3, "use_cosine_for_theta": True,  "label": "A) Same LR + Cosine (Current)"},
        {"lr_net": 1e-3, "lr_theta": 1e-2, "use_cosine_for_theta": True,  "label": "B) 10x Theta LR + Cosine"},
        {"lr_net": 1e-3, "lr_theta": 1e-3, "use_cosine_for_theta": False, "label": "C) Same LR + Constant Theta"},
        {"lr_net": 1e-3, "lr_theta": 1e-2, "use_cosine_for_theta": False, "label": "D) 10x Theta LR + Constant"},
        {"lr_net": 1e-3, "lr_theta": 5e-2, "use_cosine_for_theta": False, "label": "E) 50x Theta LR + Constant"},
    ]
    
    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {label}")
        print(f"{'='*80}")
        
        history = run_experiment(**cfg)
        
        print(f"{'Step':>5} | {'NetLoss':>12} | {'LR_theta':>10} | Precision Weights (7 tasks)")
        print("-" * 100)
        for r in history:
            w_str = " ".join(f"{w:7.3f}" for w in r['weights'])
            print(f"{r['step']:5d} | {r['net_loss']:12.4f} | {r['lr_theta_effective']:10.6f} | [{w_str}]")
        
        # Final weight distribution quality metric
        final = history[-1]
        w = final['weights']
        weight_spread = max(w) / (min(w) + 1e-10)
        print(f"\nFinal Weight Spread (max/min): {weight_spread:.2f}x")
        print(f"Final L_bar values: {[f'{l:.4f}' for l in final['l_bars']]}")
