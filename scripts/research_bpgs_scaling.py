"""
BPGS + OnlineTargetScaler Interaction Simulation
=================================================
Tests whether the combination of OnlineTargetScaler normalization and 
BPGS precision weighting creates any unintended double-scaling effects
that could harm convergence.

The question: Does normalizing targets before computing MSE, and THEN applying
BPGS precision weights, produce mathematically equivalent convergence to 
operating on raw targets? Or does the scale compression from normalization
rob BPGS of the information it needs to differentiate task difficulties?
"""
import torch
import torch.nn as nn
import math

class SimpleBPGS:
    """Simplified BPGS for testing."""
    def __init__(self, num_tasks, tau=50.0, s_min=-10.0, s_max=10.0):
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
                self.L_bar[i] = self.L_bar[i] + self.beta * (l.item() - self.L_bar[i])
                
    def precision_weights(self):
        s = self.get_s()
        return [torch.exp(-s[i]).detach() for i in range(self.num_tasks)]

class OnlineScaler:
    """Simplified OnlineTargetScaler."""
    def __init__(self, momentum=0.1):
        self.mean = 0.0
        self.var = 1.0
        self.momentum = momentum
        self.initialized = False
        
    def update(self, targets):
        m = targets.mean().item()
        v = targets.var().item() if targets.numel() > 1 else 0.0
        if not self.initialized:
            self.mean = m
            self.var = v
            self.initialized = True
        else:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * m
            self.var = (1 - self.momentum) * self.var + self.momentum * v
    
    def normalize(self, x):
        std = math.sqrt(self.var + 1e-6)
        return (x - self.mean) / std
    
    def denormalize(self, x):
        std = math.sqrt(self.var + 1e-6)
        return x * std + self.mean

def simulate_training(use_scaler=True, n_steps=200, seed=42):
    """Simulate a 2-task regression problem."""
    torch.manual_seed(seed)
    
    # Task 1: High-scale target (e.g., house prices ~500k)
    # Task 2: Low-scale target (e.g., room count ~5)
    true_scale_1 = 500000.0
    true_scale_2 = 5.0
    
    bpgs = SimpleBPGS(num_tasks=2, tau=50.0)
    
    scalers = [OnlineScaler(), OnlineScaler()] if use_scaler else [None, None]
    
    # Simple linear models
    w1 = torch.tensor([0.0], requires_grad=True)
    w2 = torch.tensor([0.0], requires_grad=True)
    net_opt = torch.optim.Adam([w1, w2], lr=0.01)
    unc_opt = torch.optim.Adam([bpgs.theta], lr=0.01)
    
    history = []
    
    for step in range(n_steps):
        # Generate random targets
        t1 = torch.randn(32) * true_scale_1 + true_scale_1
        t2 = torch.randn(32) * true_scale_2 + true_scale_2
        
        # Model predictions
        p1 = w1.expand(32) * true_scale_1
        p2 = w2.expand(32) * true_scale_2
        
        if use_scaler:
            scalers[0].update(t1)
            scalers[1].update(t2)
            t1_for_loss = scalers[0].normalize(t1)
            t2_for_loss = scalers[1].normalize(t2)
            p1_for_loss = scalers[0].normalize(p1)
            p2_for_loss = scalers[1].normalize(p2)
        else:
            t1_for_loss = t1
            t2_for_loss = t2
            p1_for_loss = p1
            p2_for_loss = p2
        
        loss1 = ((p1_for_loss - t1_for_loss) ** 2).mean()
        loss2 = ((p2_for_loss - t2_for_loss) ** 2).mean()
        
        # EMA update
        bpgs.update_ema([loss1, loss2])
        
        # Get precision weights
        weights = bpgs.precision_weights()
        
        # Network loss
        net_opt.zero_grad()
        net_loss = 0.5 * weights[0] * loss1 + 0.5 * weights[1] * loss2
        net_loss.backward()
        net_opt.step()
        
        # Uncertainty loss
        unc_opt.zero_grad()
        s = bpgs.get_s()
        l_unc = sum(0.5 * torch.exp(-s[i]) * torch.sqrt(bpgs.L_bar[i]**2 + 1e-10) + 0.5 * s[i] 
                     for i in range(2))
        l_unc.backward()
        unc_opt.step()
        
        if step % 20 == 0:
            history.append({
                'step': step,
                'loss1': loss1.item(),
                'loss2': loss2.item(),
                'L_bar': bpgs.L_bar.tolist(),
                'weights': [w.item() for w in weights],
                's_vals': bpgs.get_s().detach().tolist(),
            })
    
    return history

if __name__ == "__main__":
    print("=" * 80)
    print("EXPERIMENT 1: WITH OnlineTargetScaler (normalized losses)")
    print("=" * 80)
    h_scaled = simulate_training(use_scaler=True)
    print(f"{'Step':>5} | {'L1':>12} | {'L2':>12} | {'L_bar[0]':>12} | {'L_bar[1]':>12} | {'w[0]':>10} | {'w[1]':>10} | {'s[0]':>8} | {'s[1]':>8}")
    print("-" * 110)
    for r in h_scaled:
        print(f"{r['step']:5d} | {r['loss1']:12.4f} | {r['loss2']:12.4f} | {r['L_bar'][0]:12.4f} | {r['L_bar'][1]:12.4f} | {r['weights'][0]:10.4f} | {r['weights'][1]:10.4f} | {r['s_vals'][0]:8.4f} | {r['s_vals'][1]:8.4f}")
    
    print()
    print("=" * 80)
    print("EXPERIMENT 2: WITHOUT OnlineTargetScaler (raw losses)")
    print("=" * 80)
    h_raw = simulate_training(use_scaler=False)
    print(f"{'Step':>5} | {'L1':>12} | {'L2':>12} | {'L_bar[0]':>12} | {'L_bar[1]':>12} | {'w[0]':>10} | {'w[1]':>10} | {'s[0]':>8} | {'s[1]':>8}")
    print("-" * 110)
    for r in h_raw:
        print(f"{r['step']:5d} | {r['loss1']:12.4f} | {r['loss2']:12.4f} | {r['L_bar'][0]:12.4f} | {r['L_bar'][1]:12.4f} | {r['weights'][0]:10.4f} | {r['weights'][1]:10.4f} | {r['s_vals'][0]:8.4f} | {r['s_vals'][1]:8.4f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    final_s = h_scaled[-1]
    final_r = h_raw[-1]
    print(f"\nWith Scaler:    L_bar ratio = {final_s['L_bar'][0] / (final_s['L_bar'][1] + 1e-10):.4f}, Weight ratio = {final_s['weights'][0] / (final_s['weights'][1] + 1e-10):.4f}")
    print(f"Without Scaler: L_bar ratio = {final_r['L_bar'][0] / (final_r['L_bar'][1] + 1e-10):.4f}, Weight ratio = {final_r['weights'][0] / (final_r['weights'][1] + 1e-10):.4f}")
    print(f"\nWith Scaler:    Both L_bars should be ~1.0 (normalized) ==> BPGS treats tasks as equally noisy (GOOD)")
    print(f"Without Scaler: L_bar ratio should be ~(500000/5)^2 = 1e10 ==> BPGS completely ignores one task (BAD)")
