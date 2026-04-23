"""
BPGS Theta Schedule SOTA Comparison
=====================================
Tests PROPER scheduling strategies for the uncertainty optimizer.
The key insight: theta needs a schedule (not constant LR), but one that
maintains a HIGH FLOOR so it can track evolving task difficulties.

Candidates:
  A) Current: Cosine to near-zero (min_ratio=0.001) — BASELINE
  B) Cosine with 10% floor (min_ratio=0.1)
  C) Cosine with 30% floor (min_ratio=0.3)
  D) Inverse Square Root (Transformer-style)
  E) Cosine Warm Restarts (SGDR, T_0=500)
  F) Linear Warmup → Constant (no decay)
  G) Exponential Decay with floor (γ=0.9995, floor=10%)
"""
import torch
import math

torch.manual_seed(42)

class FullBPGSSim:
    def __init__(self, num_tasks=7, tau=50.0, s_min=-10.0, s_max=10.0):
        self.theta = torch.zeros(num_tasks, requires_grad=True)
        self.s_min, self.s_max = s_min, s_max
        self.beta = 1 - math.exp(-1/tau)
        self.L_bar = torch.ones(num_tasks)
        self.num_tasks = num_tasks
    
    def get_s(self):
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.theta)
    
    def update_ema(self, losses):
        with torch.no_grad():
            for i, l in enumerate(losses):
                if torch.isfinite(l):
                    self.L_bar[i] += self.beta * (l.item() - self.L_bar[i])
    
    def network_loss(self, raw_losses):
        s = self.get_s()
        return sum(0.5 * torch.exp(-s[i]).detach() * l for i, l in enumerate(raw_losses))
    
    def uncertainty_loss(self):
        s = self.get_s()
        return sum(0.5 * torch.exp(-s[i]) * torch.sqrt(self.L_bar[i]**2 + 1e-10) + 0.5 * s[i]
                   for i in range(self.num_tasks))

# ─── Schedule Functions ───
def cosine_schedule(step, total, warmup, min_ratio):
    if step < warmup: return float(step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

def inv_sqrt_schedule(step, warmup, d_model=256):
    """Transformer-style inverse sqrt schedule."""
    if step < warmup: return float(step) / max(1, warmup)
    return (warmup ** 0.5) / max(1, step ** 0.5)

def cosine_warm_restart(step, T_0, warmup):
    """SGDR: Cosine annealing with warm restarts."""
    if step < warmup: return float(step) / max(1, warmup)
    cycle_step = (step - warmup) % T_0
    return max(0.05, 0.5 * (1 + math.cos(math.pi * cycle_step / T_0)))

def exp_decay_with_floor(step, warmup, gamma=0.9995, floor=0.1):
    if step < warmup: return float(step) / max(1, warmup)
    return max(floor, gamma ** (step - warmup))

def constant_after_warmup(step, warmup):
    if step < warmup: return float(step) / max(1, warmup)
    return 1.0

# ─── Experiment Runner ───
def run_experiment(lr_net, lr_theta, schedule_fn_net, schedule_fn_theta, total_steps=2000, warmup=100, label=""):
    torch.manual_seed(42)
    task_complexities = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    num_tasks = len(task_complexities)
    bpgs = FullBPGSSim(num_tasks=num_tasks)
    models = [torch.tensor([0.5], requires_grad=True) for _ in range(num_tasks)]
    opt_net = torch.optim.AdamW(models, lr=lr_net, weight_decay=0.0)
    opt_unc = torch.optim.AdamW([bpgs.theta], lr=lr_theta, weight_decay=0.0)
    
    history = []
    for step in range(total_steps):
        lr_mult = schedule_fn_net(step)
        for pg in opt_net.param_groups: pg['lr'] = lr_net * lr_mult
        lr_mult_t = schedule_fn_theta(step)
        for pg in opt_unc.param_groups: pg['lr'] = lr_theta * lr_mult_t
        
        raw_losses = []
        for i, c in enumerate(task_complexities):
            target = torch.randn(32) * c + c
            pred = models[i] * c
            raw_losses.append(((pred - target) ** 2).mean())
        
        bpgs.update_ema(raw_losses)
        opt_net.zero_grad()
        net_loss = bpgs.network_loss(raw_losses)
        net_loss.backward()
        opt_net.step()
        
        opt_unc.zero_grad()
        unc_loss = bpgs.uncertainty_loss()
        unc_loss.backward()
        opt_unc.step()
        
        if step % 250 == 0 or step == total_steps - 1:
            s_vals = bpgs.get_s().detach()
            weights = [math.exp(-s_vals[i].item()) for i in range(num_tasks)]
            history.append({
                'step': step, 'net_loss': net_loss.item(),
                'weights': weights, 'l_bars': bpgs.L_bar.tolist(),
                'lr_theta_eff': opt_unc.param_groups[0]['lr'],
            })
    return history

if __name__ == "__main__":
    TOTAL = 2000
    WARMUP = 100
    LR_NET = 1e-3
    LR_THETA = 1e-2  # 10x network LR (proven optimal from previous research)
    
    configs = [
        {
            "label": "A) BASELINE: Cosine → near-zero (min=0.001)",
            "lr_theta": LR_NET,  # Current: same LR as network
            "sched_theta": lambda s: cosine_schedule(s, TOTAL, WARMUP, 0.001),
        },
        {
            "label": "B) 10x LR + Cosine w/ 10% floor",
            "lr_theta": LR_THETA,
            "sched_theta": lambda s: cosine_schedule(s, TOTAL, WARMUP, 0.1),
        },
        {
            "label": "C) 10x LR + Cosine w/ 30% floor",
            "lr_theta": LR_THETA,
            "sched_theta": lambda s: cosine_schedule(s, TOTAL, WARMUP, 0.3),
        },
        {
            "label": "D) 10x LR + Inverse Sqrt (Transformer)",
            "lr_theta": LR_THETA,
            "sched_theta": lambda s: inv_sqrt_schedule(s, WARMUP),
        },
        {
            "label": "E) 10x LR + Cosine Warm Restarts (T_0=500)",
            "lr_theta": LR_THETA,
            "sched_theta": lambda s: cosine_warm_restart(s, 500, WARMUP),
        },
        {
            "label": "F) 10x LR + Warmup → Constant",
            "lr_theta": LR_THETA,
            "sched_theta": lambda s: constant_after_warmup(s, WARMUP),
        },
        {
            "label": "G) 10x LR + ExpDecay (γ=0.9995, floor=10%)",
            "lr_theta": LR_THETA,
            "sched_theta": lambda s: exp_decay_with_floor(s, WARMUP, 0.9995, 0.1),
        },
    ]
    
    sched_net = lambda s: cosine_schedule(s, TOTAL, WARMUP, 1e-6 / LR_NET)
    
    results_summary = []
    
    for cfg in configs:
        label = cfg["label"]
        print(f"\n{'='*90}")
        print(f"  {label}")
        print(f"{'='*90}")
        
        history = run_experiment(
            lr_net=LR_NET, lr_theta=cfg["lr_theta"],
            schedule_fn_net=sched_net,
            schedule_fn_theta=cfg["sched_theta"],
            total_steps=TOTAL, warmup=WARMUP, label=label,
        )
        
        print(f"{'Step':>5} | {'NetLoss':>10} | {'LR_θ':>10} | Precision Weights")
        print("-" * 90)
        for r in history:
            w_str = " ".join(f"{w:7.3f}" for w in r['weights'])
            print(f"{r['step']:5d} | {r['net_loss']:10.4f} | {r['lr_theta_eff']:10.6f} | [{w_str}]")
        
        final = history[-1]
        w = final['weights']
        spread = max(w) / (min(w) + 1e-10)
        
        # Compute average loss over last 500 steps (stability metric)
        late_losses = [h['net_loss'] for h in history if h['step'] >= 1500]
        avg_late = sum(late_losses) / max(1, len(late_losses))
        
        results_summary.append({
            'label': label,
            'final_loss': final['net_loss'],
            'avg_late_loss': avg_late,
            'spread': spread,
            'final_lr': final['lr_theta_eff'],
        })
    
    # ─── Summary Comparison ───
    print(f"\n{'='*90}")
    print(f"  FINAL COMPARISON (sorted by average late-training loss)")
    print(f"{'='*90}")
    results_summary.sort(key=lambda x: x['avg_late_loss'])
    
    print(f"{'Config':<50} | {'Avg Late Loss':>13} | {'Final Loss':>10} | {'Spread':>10} | {'Final LR_θ':>10}")
    print("-" * 105)
    for r in results_summary:
        print(f"{r['label']:<50} | {r['avg_late_loss']:13.4f} | {r['final_loss']:10.4f} | {r['spread']:10.1f}x | {r['final_lr']:10.6f}")
