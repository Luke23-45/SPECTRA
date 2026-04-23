import math


def evaluate_ema(decay, tau_formula_name, tau_formula_func, n_steps=500):
    tau = tau_formula_func(decay)
    
    # In BPGS, beta = 1 - exp(-1/tau)
    beta = 1.0 - math.exp(-1.0 / tau)
    
    # Simulate step response
    # True signal is 0 for t < 100, 1 for t >= 100
    # Add noise? Let's just do a clean step response to see the time constant
    history = []
    L_bar = 0.0
    for t in range(n_steps):
        loss = 1.0 if t >= 100 else 0.0
        
        # update formula used in BPGS
        L_bar = L_bar + beta * (loss - L_bar)
        history.append(L_bar)
        
    return tau, beta, history

decay_target = 0.99

formulas = {
    "Current Buggy": lambda d: -1.0 / math.log(1.0 - d + 1e-8), 
    "SOTA Exact": lambda d: -1.0 / math.log(d),
    "Euler Approx": lambda d: 1.0 / (1.0 - d)  # alpha = 1/tau -> tau = 1/alpha = 1/(1-decay)
}

results = {}
for name, func in formulas.items():
    if name == "Euler Approx" and decay_target == 1.0:
        continue # avoid div by zero
    tau, beta, hist = evaluate_ema(decay_target, name, func)
    results[name] = {"tau": tau, "beta": beta, "hist": hist}

print(f"Target Decay (History Weight): {decay_target}")
print("-" * 50)
for name, res in results.items():
    print(f"[{name}]")
    print(f"  Calculated Tau:  {res['tau']:.4f}")
    print(f"  Resulting Beta (Current Batch Weight): {res['beta']:.6f}")
    print(f"  Effective Decay (1 - Beta): {1.0 - res['beta']:.6f}")
    
    # Calculate time to reach 63.2% (1 - 1/e) of the step
    # Step happens at t=100. Target is 1.0 * (1 - 1/e) = ~0.632
    time_to_63 = -1
    for t in range(100, 500):
        if res['hist'][t] >= 1.0 - math.exp(-1.0):
            time_to_63 = t - 100
            break
    
    print(f"  Time constant (steps to 63.2%): {time_to_63}")
    print()

