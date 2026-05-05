import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure spectra can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spectra.core.bpgs import BPGS

def generate_synthetic_loss(task_type, step, max_steps, base_scale, noise_level):
    """
    Generates a synthetic scalar loss value representing a batch loss at a given training step.
    """
    progress = step / max_steps
    
    if task_type == 'cross_entropy':
        # CE typically starts high (e.g., 2-4) and decays exponentially. 
        # Prone to large spikes if the model is uncertain.
        mean = base_scale * np.exp(-3 * progress)
        noise = np.random.normal(0, noise_level * mean)
        val = mean + noise + np.random.exponential(scale=noise_level*0.5) # random spikes
        return max(0.01, val)
        
    elif task_type == 'masked_l1':
        # L1 is linear. Starts at some value and decreases linearly.
        mean = base_scale * (1 - 0.8 * progress)
        noise = np.random.normal(0, noise_level * mean)
        val = mean + noise
        return max(0.001, val)
        
    elif task_type == 'cosine':
        # Cosine distance is bounded [0, 2]. Typically starts around 1.0 and drops.
        mean = base_scale * (1 - 0.9 * progress)
        noise = np.random.uniform(-noise_level, noise_level)
        val = mean + noise
        return max(0.001, min(2.0, val))
        
    else:
        # Generic fallback
        return max(0.01, base_scale * (1 - progress) + np.random.normal(0, noise_level))

def run_stress_test(scenario_name, tasks, steps=2000, lr=0.05, use_batch_surrogate=False, use_temperature_equalization=False, temperature=2.0):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    num_tasks = len(tasks)
    
    # Initialize BPGS
    bpgs = BPGS(num_tasks=num_tasks)
    optimizer = torch.optim.Adam(bpgs.parameters(), lr=lr)
    
    # Running-average trackers for each task
    avg_losses = np.ones(num_tasks) * -1.0 
    
    history = {
        's': np.zeros((steps, num_tasks)),
        'weights': np.zeros((steps, num_tasks)),
        'losses': np.zeros((steps, num_tasks)),
        'total_u_loss': np.zeros(steps)
    }
    
    for step in range(steps):
        # Generate mock loss tensors
        raw_losses = []
        surrogate_losses = []
        
        for i, task in enumerate(tasks):
            val = generate_synthetic_loss(task['type'], step, steps, task['scale'], task['noise'])
            loss_tensor = torch.tensor(val, dtype=torch.float32, requires_grad=True)
            raw_losses.append(loss_tensor)
            history['losses'][step, i] = val
            
            # Running-average update
            if avg_losses[i] < 0:
                avg_losses[i] = val
            else:
                avg_losses[i] = 0.9 * avg_losses[i] + 0.1 * val

            # Create Surrogate Loss for Path B
            surrogate_losses.append(loss_tensor)
        
        # Path B: Stateless Batch Normalization
        if use_batch_surrogate:
            # Calculate the mean of the raw losses in this specific batch
            batch_mean = sum([l.item() for l in raw_losses]) / num_tasks
            batch_surrogate_losses = []
            for loss_tensor in raw_losses:
                # Divide by the batch mean, effectively equalizing their scales statelessly
                normalized_tensor = loss_tensor / max(1e-6, batch_mean)
                batch_surrogate_losses.append(normalized_tensor)
            surrogate_losses = batch_surrogate_losses
            
        optimizer.zero_grad()
        
        # B-PGS uncertainty loss (updates s/theta) using chosen losses
        u_loss = bpgs.uncertainty_loss(surrogate_losses)
        u_loss.backward()
        optimizer.step()
        
        # Record stats
        stats = bpgs.get_task_stats()
        history['total_u_loss'][step] = u_loss.item()
        
        # Apply Temperature Equalization to the Weights
        raw_s = np.array([stats[f'bpgs/log_var_{i}'] for i in range(num_tasks)])
        if use_temperature_equalization:
            # W = num_tasks * Softmax(-s / T)
            logits = -raw_s / temperature
            exp_logits = np.exp(logits - np.max(logits)) # stable softmax
            softmax_weights = exp_logits / np.sum(exp_logits)
            final_weights = num_tasks * softmax_weights
        else:
            final_weights = np.array([stats[f'bpgs/weight_{i}'] for i in range(num_tasks)])
            
        for i in range(num_tasks):
            history['s'][step, i] = raw_s[i]
            history['weights'][step, i] = final_weights[i]
            
    print("Final State:")
    for i, task in enumerate(tasks):
        print(f"Task {i} ({task['type']}): Loss={history['losses'][-1, i]:.4f}, "
              f"s={history['s'][-1, i]:.4f}, Weight={history['weights'][-1, i]:.4f}")
        
    return history

def plot_scenario(scenario_name, tasks, history, out_dir):
    steps = len(history['total_u_loss'])
    x = np.arange(steps)
    num_tasks = len(tasks)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Raw Losses
    for i in range(num_tasks):
        axs[0].plot(x, history['losses'][:, i], label=f"Task {i}: {tasks[i]['type']}")
    axs[0].set_title(f"{scenario_name} - Synthetic Raw Losses")
    axs[0].set_ylabel("Loss Value")
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: B-PGS Parameters (s)
    for i in range(num_tasks):
        axs[1].plot(x, history['s'][:, i], label=f"Task {i} (s)")
    axs[1].set_title("B-PGS Log-Variance (s)")
    axs[1].set_ylabel("s value")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Resulting Weights
    for i in range(num_tasks):
        axs[2].plot(x, history['weights'][:, i], label=f"Task {i} weight")
    axs[2].set_title("B-PGS Precision Weights (exp(-s))")
    axs[2].set_ylabel("Weight")
    axs[2].set_xlabel("Training Step")
    axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scenario_name.replace(' ', '_')}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'stress_test_results')
    
    # Scenario 1: Homogeneous (All Cross Entropy)
    tasks_homo = [
        {'type': 'cross_entropy', 'scale': 3.0, 'noise': 0.1},
        {'type': 'cross_entropy', 'scale': 3.0, 'noise': 0.1},
        {'type': 'cross_entropy', 'scale': 3.0, 'noise': 0.1},
    ]
    hist_homo = run_stress_test("Homogeneous Tasks", tasks_homo)
    plot_scenario("Homogeneous Tasks", tasks_homo, hist_homo, out_dir)
    
    # Scenario 2: Heterogeneous (NYUv2 Simulation)
    tasks_hetero = [
        {'type': 'cross_entropy', 'scale': 3.0, 'noise': 0.1}, # Task 0: Seg
        {'type': 'masked_l1', 'scale': 1.0, 'noise': 0.05},    # Task 1: Depth
        {'type': 'cosine', 'scale': 0.5, 'noise': 0.05},       # Task 2: Normals
    ]
    hist_hetero = run_stress_test("Heterogeneous Tasks", tasks_hetero)
    plot_scenario("Heterogeneous Tasks", tasks_hetero, hist_hetero, out_dir)
    
    # Scenario 3: Scale Extreme Mismatch
    tasks_extreme = [
        {'type': 'cross_entropy', 'scale': 100.0, 'noise': 0.2}, # Huge scale
        {'type': 'cosine', 'scale': 0.01, 'noise': 0.01},        # Tiny scale
    ]
    hist_extreme = run_stress_test("Extreme Mismatch", tasks_extreme)
    plot_scenario("Extreme Mismatch", tasks_extreme, hist_extreme, out_dir)

    # --- Path B Prototyping (Stateless) ---
    
    # Scenario 5: Heterogeneous with Batch Surrogate
    hist_hetero_batch = run_stress_test("Heterogeneous Tasks (Batch Surrogate)", tasks_hetero, use_batch_surrogate=True)
    plot_scenario("Heterogeneous_Tasks_Batch_Surrogate", tasks_hetero, hist_hetero_batch, out_dir)
    
    # Scenario 6: Extreme Mismatch with Batch Surrogate
    hist_extreme_batch = run_stress_test("Extreme Mismatch (Batch Surrogate)", tasks_extreme, use_batch_surrogate=True)
    plot_scenario("Extreme_Mismatch_Batch_Surrogate", tasks_extreme, hist_extreme_batch, out_dir)

    # --- Path B Prototyping (Softmax Temperature) ---
    # Scenario 7: Heterogeneous Tasks (Temperature Equalization)
    hist_hetero_temp = run_stress_test("Heterogeneous Tasks (Temp Eq)", tasks_hetero, use_temperature_equalization=True, temperature=2.0)
    plot_scenario("Heterogeneous_Tasks_Temp_Eq", tasks_hetero, hist_hetero_temp, out_dir)
    
    # Scenario 8: Extreme Mismatch (Temperature Equalization)
    hist_extreme_temp = run_stress_test("Extreme Mismatch (Temp Eq)", tasks_extreme, use_temperature_equalization=True, temperature=2.0)
    plot_scenario("Extreme_Mismatch_Temp_Eq", tasks_extreme, hist_extreme_temp, out_dir)

if __name__ == '__main__':
    main()
