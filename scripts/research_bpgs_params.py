"""
B-PGS Parameter Optimization Research — Empirical Head-to-Head Comparison
=========================================================================
Tests every tunable parameter and design choice in bpgs.py against alternatives.

Each experiment simulates 2000 training steps where 7 tasks have different
loss trajectories (rapidly decreasing, slowly decreasing, oscillating, etc.)
The "winner" is the configuration with the lowest final aggregate loss
AND the most stable convergence (lowest late-training variance).

Parameters tested:
  A: s_bounds     — [-10,10] vs [-5,5] vs [-20,20] vs asymmetric [-5,15]
  B: tau          — 10 vs 50 vs 100 vs adaptive-per-task
  C: eps (R_eps)  — 1e-5 vs 1e-3 vs 1e-7
  D: diffeomorphism — sigmoid vs scaled_tanh vs arctan
  E: L_bar_init   — 1.0 vs first-batch vs BEMA bias correction
  F: prior_var    — None vs 1.0 vs 10.0 vs 100.0
  G: the 0.5 coeff— 0.5 (standard Kendall) vs 1.0 vs learned
"""

import math
import torch
import torch.nn as nn
import sys
import os

# ──────────────────────────────────────────────────────────────────────
# Simulated multi-task loss trajectories (ground truth)
# ──────────────────────────────────────────────────────────────────────
def generate_loss_trajectories(n_steps=2000, n_tasks=7, seed=42):
    """Generate realistic per-task loss curves for simulation."""
    torch.manual_seed(seed)
    t = torch.arange(n_steps, dtype=torch.float32)
    
    trajectories = {}
    # Task 0: Rapid exponential decay (easy task)
    trajectories[0] = 5.0 * torch.exp(-t / 200) + 0.1 + 0.05 * torch.randn(n_steps)
    # Task 1: Slow linear decay (hard task)
    trajectories[1] = 10.0 - 4.0 * (t / n_steps) + 0.2 * torch.randn(n_steps)
    # Task 2: Near-zero (solved task) — tests R_eps precision
    trajectories[2] = 0.01 * torch.exp(-t / 100) + 0.001 + 0.001 * torch.randn(n_steps)
    # Task 3: Oscillating (unstable task)
    trajectories[3] = 2.0 + 1.0 * torch.sin(t / 50) + 0.3 * torch.randn(n_steps)
    # Task 4: Plateau then drop (phase transition)
    trajectories[4] = torch.where(t < 1000, torch.tensor(3.0), 3.0 * torch.exp(-(t - 1000) / 300))
    trajectories[4] = trajectories[4] + 0.1 * torch.randn(n_steps)
    # Task 5: Very high scale (scale-gap stress test)
    trajectories[5] = 1000.0 * torch.exp(-t / 500) + 50.0 + 5.0 * torch.randn(n_steps)
    # Task 6: BCE-scale (bounded ~0.69)
    trajectories[6] = 0.693 - 0.3 * (t / n_steps) + 0.02 * torch.randn(n_steps)
    
    # Clamp all to positive
    for k in trajectories:
        trajectories[k] = trajectories[k].clamp(min=1e-8)
    
    return trajectories

# ──────────────────────────────────────────────────────────────────────
# R_eps variants
# ──────────────────────────────────────────────────────────────────────
def R_eps(x, eps=1e-5):
    return torch.sqrt(x.pow(2) + eps * eps)

# ──────────────────────────────────────────────────────────────────────
# Diffeomorphism variants
# ──────────────────────────────────────────────────────────────────────
def sigmoid_chart(theta, s_min, s_max):
    """Standard sigmoid: s = s_min + (s_max - s_min) * σ(θ)"""
    return s_min + (s_max - s_min) * torch.sigmoid(theta)

def scaled_tanh_chart(theta, s_min, s_max):
    """Tanh: s = (s_min + s_max)/2 + (s_max - s_min)/2 * tanh(θ)
    Advantage: zero-centered derivative, max gradient at θ=0"""
    mid = (s_min + s_max) / 2.0
    half_range = (s_max - s_min) / 2.0
    return mid + half_range * torch.tanh(theta)

def arctan_chart(theta, s_min, s_max):
    """Arctan: s = (s_min + s_max)/2 + (s_max - s_min)/π * arctan(θ)
    Advantage: heavier tails, slower saturation"""
    mid = (s_min + s_max) / 2.0
    half_range = (s_max - s_min) / 2.0
    return mid + half_range * (2.0 / math.pi) * torch.atan(theta)

# ──────────────────────────────────────────────────────────────────────
# BPGS Simulation Core
# ──────────────────────────────────────────────────────────────────────
class BPGSSimulator:
    """Simulates B-PGS weight allocation for given loss trajectories."""
    
    def __init__(
        self,
        n_tasks=7,
        s_min=-10.0, s_max=10.0,
        tau=50.0,
        eps=1e-5,
        chart_fn=sigmoid_chart,
        l_bar_init=1.0,
        use_bema=False,  # Bias-corrected EMA
        prior_var=None,
        coeff=0.5,       # The 0.5 coefficient in the loss
        lr_theta=0.01,
    ):
        self.n_tasks = n_tasks
        self.s_min = s_min
        self.s_max = s_max
        self.eps = eps
        self.chart_fn = chart_fn
        self.prior_var = prior_var
        self.coeff = coeff
        self.lr_theta = lr_theta
        self.use_bema = use_bema
        
        # Exact EMA
        self.tau = tau
        self.beta = 1.0 - math.exp(-1.0 / tau)
        
        # State
        self.theta = torch.zeros(n_tasks, requires_grad=True)
        self.L_bar = torch.full((n_tasks,), l_bar_init)
        self.step_count = 0
        
        # Optimizer for theta
        self.opt = torch.optim.Adam([self.theta], lr=lr_theta)
    
    def step(self, losses):
        """One optimization step given current batch losses."""
        self.step_count += 1
        
        # 1. Update EMA
        with torch.no_grad():
            for i in range(self.n_tasks):
                loss_val = losses[i].item()
                if not math.isfinite(loss_val):
                    continue
                old_val = self.L_bar[i].item()
                new_val = old_val + self.beta * (loss_val - old_val)
                
                # BEMA bias correction (like Adam's bias correction)
                if self.use_bema:
                    correction = 1.0 - (1.0 - self.beta) ** self.step_count
                    new_val = new_val / correction
                
                self.L_bar[i] = new_val
        
        # 2. Compute uncertainty loss and update theta
        self.opt.zero_grad()
        s_values = self.chart_fn(self.theta, self.s_min, self.s_max)
        
        unc_loss = torch.tensor(0.0)
        for i in range(self.n_tasks):
            s_i = s_values[i]
            w_i = torch.exp(-s_i)
            r_val = R_eps(self.L_bar[i].detach(), eps=self.eps)
            unc_loss = unc_loss + self.coeff * w_i * r_val + self.coeff * s_i
        
        if self.prior_var is not None:
            unc_loss = unc_loss + (0.5 / self.prior_var) * self.theta.pow(2).sum()
        
        unc_loss.backward()
        self.opt.step()
        
        # 3. Compute weighted loss (what the network would see)
        with torch.no_grad():
            s_values = self.chart_fn(self.theta, self.s_min, self.s_max)
            weights = torch.exp(-s_values)
            weighted_sum = sum(0.5 * weights[i] * losses[i] for i in range(self.n_tasks))
        
        return weighted_sum.item(), weights.tolist()

def run_experiment(config_name, **kwargs):
    """Run a single experiment and return metrics."""
    trajectories = generate_loss_trajectories()
    n_steps = 2000
    n_tasks = 7
    
    sim = BPGSSimulator(n_tasks=n_tasks, **kwargs)
    
    weighted_losses = []
    weight_history = []
    
    for t in range(n_steps):
        losses = [trajectories[i][t] for i in range(n_tasks)]
        wl, weights = sim.step(losses)
        weighted_losses.append(wl)
        if t % 100 == 0:
            weight_history.append(weights)
    
    # Metrics
    late_losses = weighted_losses[1500:]  # Last 25%
    avg_late = sum(late_losses) / len(late_losses)
    var_late = sum((x - avg_late)**2 for x in late_losses) / len(late_losses)
    final_weights = weight_history[-1]
    
    # Check for NaN/Inf
    has_nan = any(not math.isfinite(x) for x in weighted_losses)
    
    return {
        'name': config_name,
        'avg_late_loss': avg_late,
        'var_late_loss': var_late,
        'final_weights': [f'{w:.4f}' for w in final_weights],
        'has_nan': has_nan,
        'final_theta': sim.theta.detach().tolist(),
    }

# ──────────────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

experiments = {
    # ── A: s_bounds ──
    "A1_bounds_[-10,10]":  dict(s_min=-10.0, s_max=10.0),
    "A2_bounds_[-5,5]":    dict(s_min=-5.0,  s_max=5.0),
    "A3_bounds_[-20,20]":  dict(s_min=-20.0, s_max=20.0),
    "A4_bounds_[-5,15]":   dict(s_min=-5.0,  s_max=15.0),  # Asymmetric
    
    # ── B: tau ──
    "B1_tau_10":  dict(tau=10.0),
    "B2_tau_50":  dict(tau=50.0),
    "B3_tau_100": dict(tau=100.0),
    "B4_tau_200": dict(tau=200.0),
    
    # ── C: eps ──
    "C1_eps_1e-3": dict(eps=1e-3),
    "C2_eps_1e-5": dict(eps=1e-5),
    "C3_eps_1e-7": dict(eps=1e-7),
    
    # ── D: diffeomorphism ──
    "D1_sigmoid":     dict(chart_fn=sigmoid_chart),
    "D2_scaled_tanh": dict(chart_fn=scaled_tanh_chart),
    "D3_arctan":      dict(chart_fn=arctan_chart),
    
    # ── E: L_bar init ──
    "E1_lbar_1.0":     dict(l_bar_init=1.0),
    "E2_lbar_0.1":     dict(l_bar_init=0.1),
    "E3_lbar_10.0":    dict(l_bar_init=10.0),
    "E4_bema":         dict(l_bar_init=0.0, use_bema=True),  # BEMA bias correction
    
    # ── F: prior_var ──
    "F1_no_prior":     dict(prior_var=None),
    "F2_prior_1.0":    dict(prior_var=1.0),
    "F3_prior_10.0":   dict(prior_var=10.0),
    "F4_prior_100.0":  dict(prior_var=100.0),
    
    # ── G: coefficient ──
    "G1_coeff_0.5":    dict(coeff=0.5),
    "G2_coeff_1.0":    dict(coeff=1.0),
}

# ──────────────────────────────────────────────────────────────────────
# MAIN — Run all experiments and rank
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("B-PGS PARAMETER OPTIMIZATION — EMPIRICAL HEAD-TO-HEAD COMPARISON")
    print("=" * 80)
    
    all_results = []
    
    # Run experiments grouped by parameter
    groups = {}
    for name, kwargs in experiments.items():
        group = name.split("_")[0]
        if group not in groups:
            groups[group] = []
        groups[group].append((name, kwargs))
    
    for group_id in sorted(groups.keys()):
        group = groups[group_id]
        group_letter = group_id[0]
        print(f"\n{'─' * 80}")
        print(f"GROUP {group_id}: Testing {len(group)} configurations")
        print(f"{'─' * 80}")
        
        group_results = []
        for name, kwargs in group:
            try:
                result = run_experiment(name, **kwargs)
                group_results.append(result)
                status = "🚨 NaN!" if result['has_nan'] else "✅"
                print(f"  {status} {name:30s}  AvgLate={result['avg_late_loss']:12.4f}  "
                      f"VarLate={result['var_late_loss']:12.4f}")
            except Exception as e:
                print(f"  ❌ {name:30s}  CRASHED: {e}")
                group_results.append({'name': name, 'avg_late_loss': float('inf'), 
                                      'var_late_loss': float('inf'), 'has_nan': True})
        
        # Rank within group
        valid = [r for r in group_results if not r.get('has_nan', True) and math.isfinite(r['avg_late_loss'])]
        if valid:
            winner = min(valid, key=lambda r: r['avg_late_loss'])
            print(f"\n  🏆 GROUP {group_id} WINNER: {winner['name']}  "
                  f"(AvgLate={winner['avg_late_loss']:.4f})")
        
        all_results.extend(group_results)
    
    # ── Final ranking across all experiments ──
    print(f"\n{'=' * 80}")
    print("FINAL RANKING — ALL EXPERIMENTS")
    print(f"{'=' * 80}")
    
    valid_results = [r for r in all_results if not r.get('has_nan', True) and math.isfinite(r['avg_late_loss'])]
    valid_results.sort(key=lambda r: r['avg_late_loss'])
    
    print(f"\n{'Rank':<6} {'Config':<35} {'AvgLate':>12} {'VarLate':>12}")
    print("─" * 70)
    for i, r in enumerate(valid_results[:15], 1):
        print(f"  {i:<4} {r['name']:<35} {r['avg_late_loss']:12.4f} {r['var_late_loss']:12.4f}")
    
    # ── Per-group winners summary ──
    print(f"\n{'=' * 80}")
    print("PER-GROUP WINNERS — RECOMMENDED PARAMETERS")
    print(f"{'=' * 80}")
    for group_id in sorted(groups.keys()):
        group_results = [r for r in all_results if r['name'].startswith(group_id)]
        valid = [r for r in group_results if not r.get('has_nan', True) and math.isfinite(r['avg_late_loss'])]
        if valid:
            winner = min(valid, key=lambda r: r['avg_late_loss'])
            print(f"  {group_id}: {winner['name']} (AvgLate={winner['avg_late_loss']:.4f})")
