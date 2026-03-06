import math
import numpy as np

def simulate_ema(tau, n_steps=300, init_val=1.0, true_loss_start=10.0, use_bias_correction=False):
    beta = 1.0 - math.exp(-1.0 / tau)
    
    L_bar = init_val
    history = []
    
    for t in range(1, n_steps + 1):
        # Simulate loss that drops exponentially from true_loss_start down to 0.1
        current_loss = 0.1 + (true_loss_start - 0.1) * math.exp(-t / 50.0)
        
        # Add some noise
        # current_loss += np.random.normal(0, 0.5)
        current_loss = max(0.01, current_loss)
        
        if use_bias_correction:
            # Standard EMA
            L_bar = L_bar + beta * (current_loss - L_bar)
            # Bias correction (like Adam)
            # Because beta is the updates weight, the decay is (1 - beta)
            decay = 1.0 - beta
            correction_factor = 1.0 - math.pow(decay, t)
            L_bar_hat = L_bar / correction_factor
            history.append(L_bar_hat)
        else:
            L_bar = L_bar + beta * (current_loss - L_bar)
            history.append(L_bar)
            
    return history

if __name__ == "__main__":
    n_steps = 200
    tau = 50.0
    
    # 1. Standard BPGS EMA (Init=1.0)
    std_hist = simulate_ema(tau, n_steps, init_val=1.0, use_bias_correction=False)
    
    # 2. Bias-Corrected EMA (Init=0.0)
    bias_hist = simulate_ema(tau, n_steps, init_val=0.0, use_bias_correction=True)
    
    # 3. True Loss
    true_hist = [0.1 + (10.0 - 0.1) * math.exp(-t / 50.0) for t in range(1, n_steps + 1)]
    
    print("Step | True Loss | Standard EMA (Init 1) | Bias-Corrected (Init 0)")
    print("-" * 65)
    for t in [1, 5, 10, 20, 50, 100, 199]:
        idx = t - 1
        print(f"{t:4d} | {true_hist[idx]:9.4f} | {std_hist[idx]:19.4f} | {bias_hist[idx]:23.4f}")
        
    # Analysis:
    # Does the Init=1.0 heavily bias the early steps downward if true loss is 10.0?
    # Yes. Standard EMA at Step 5 is ~1.17 while True is 9.0. It takes ~50 steps to catch up.
    # Meanwhile, Bias Corrected jumps immediately to the true magnitude.
