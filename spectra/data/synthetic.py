"""
spectra/data/synthetic.py
-------------------------
Synthetic N-Task Multi-Task Learning Dataset.

Generates a controlled environment with extreme scale gaps between tasks
(e.g., 6000:1 MSE vs BCE). Used as the fast-iteration benchmark.

Data Generation:
    X ~ N(0, I) ∈ R^{N × input_dim}
    Z = X @ W_shared ∈ R^{N × hidden_dim}    (shared representation)

    MSE targets: y_i = Z @ w_i + offset_i      (continuous)
    BCE targets: y_j = sigmoid(Z @ w_j) > 0.5  (binary)
"""

import torch
from torch.utils.data import Dataset, default_collate
from typing import Dict, List, Optional


class SyntheticMTLDataset(Dataset):
    """
    Synthetic multi-task dataset with configurable scale gaps.

    Default configuration (7 tasks):
        Task 0: MSE, scale ~3000 (dominant)
        Task 1: MSE, scale ~10 (medium)
        Task 2: MSE, scale ~0.5 (small)
        Task 3-5: BCE (binary classification)
        Task 6: MSE, scale ~4 (auxiliary)

    Args:
        n_samples: Number of samples to generate.
        input_dim: Dimension of input features.
        hidden_dim: Dimension of shared hidden representation.
        task_configs: List of dicts, each with 'type' ('mse' or 'bce'),
                      'scale' (float), and 'offset' (float).
        seed: Random seed for data generation.
    """

    DEFAULT_TASKS = [
        {"name": "mse_high",  "type": "mse", "scale": 50.0, "offset": 3000.0},
        {"name": "mse_med",   "type": "mse", "scale": 5.0,  "offset": 10.0},
        {"name": "mse_low",   "type": "mse", "scale": 0.5,  "offset": 0.0},
        {"name": "bce_0",     "type": "bce", "scale": 1.0,  "offset": 0.0},
        {"name": "bce_1",     "type": "bce", "scale": 1.0,  "offset": 0.0},
        {"name": "bce_2",     "type": "bce", "scale": 1.0,  "offset": 0.0},
        {"name": "mse_aux",   "type": "mse", "scale": 2.0,  "offset": 0.0},
    ]

    def __init__(
        self,
        n_samples: int = 10000,
        input_dim: int = 20,
        hidden_dim: int = 64,
        task_configs: Optional[List[Dict]] = None,
        seed: int = 42,
        mapping_seed: int = 42,
    ):
        super().__init__()
        self.task_configs = task_configs or self.DEFAULT_TASKS

        # [SOTA FIX]: Decouple Data Generation from Mapping Generation.
        # Otherwise passing `seed=seed+1` for validation inadvertently changes the 
        # actual target mapping network, making validation loss mathematically diverge.
        gen_data = torch.Generator().manual_seed(seed)
        
        # [SOTA FIX]: Generator Collision Leakage. 
        # If mapping_seed == seed, the first N elements of X will perfectly mirror W_shared.
        # We must mathematically offset the generator states to guarantee orthogonality.
        gen_mapping = torch.Generator().manual_seed(mapping_seed + 1048576)

        # [SOTA FIX]: Generate highly non-linear shared features to induce gradient conflicts
        # A purely linear mapping is trivial and won't trigger PCGrad or B-PGS surgeries.
        self.X = torch.randn(n_samples, input_dim, generator=gen_data)
        
        W_shared1 = torch.randn(input_dim, hidden_dim, generator=gen_mapping) * 0.1
        W_shared2 = torch.randn(input_dim, hidden_dim, generator=gen_mapping) * 0.1
        
        # Two-stream topology (Tanh + ReLU) forces a non-convex representation space
        Z = torch.tanh(self.X @ W_shared1) + torch.relu(self.X @ W_shared2)  # [N, hidden_dim]

        # Generate per-task targets with explicit Aleatoric Noise (Homoscedastic uncertainty)
        # Probabilistic algorithms (Kendall, UWSO) will crash or collapse if the data is 100% deterministic
        self.targets = {}
        for cfg in self.task_configs:
            W_task = torch.randn(hidden_dim, 1, generator=gen_mapping) * cfg["scale"]

            if cfg["type"] == "mse":
                y = Z @ W_task + cfg["offset"]  # [N, 1]
                # Inject 10% structural Gaussian noise (Irreducible Error)
                noise = torch.randn(y.size(), generator=gen_data, dtype=y.dtype, device=y.device) * (cfg["scale"] * 0.1)
                self.targets[cfg["name"]] = (y + noise).squeeze(-1)

            elif cfg["type"] == "bce":
                # Ensure the configurable offset is actually used for classification imbalance
                logits = Z @ W_task + cfg["offset"]  # [N, 1]
                
                # [SOTA FIX]: Probabilistic targets.
                # A hard margin `y = (logits > 0)` creates a deterministic step function.
                # Optimizing BCE on a deterministic step function pushes network weights to infinity, 
                # causing gradient magnitude explosion. This physically destroys gradient-variance 
                # probabilistic weighters like B-PGS and Kendall. We must use proper Bernoulli 
                # sampling to create a calibrated, finite-weight stationary optimum.
                probs = torch.sigmoid(logits)
                y = torch.bernoulli(probs, generator=gen_data)
                
                self.targets[cfg["name"]] = y.squeeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input": self.X[idx],
            "targets": {name: self.targets[name][idx] for name in self.targets},
        }

    @property
    def task_names(self) -> List[str]:
        return [cfg["name"] for cfg in self.task_configs]

    @property
    def num_tasks(self) -> int:
        return len(self.task_configs)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate for nested dict structure.

        Default collate handles this correctly for PyTorch >= 1.9,
        but this explicit version is safer for edge cases and ensures
        the structure is exactly {"input": [B, D], "targets": {name: [B]}}.
        """
        inputs = torch.stack([item["input"] for item in batch])
        target_keys = batch[0]["targets"].keys()
        targets = {
            key: torch.stack([item["targets"][key] for item in batch])
            for key in target_keys
        }
        return {"input": inputs, "targets": targets}


# =====================================================================
# STANDALONE VERIFICATION
# =====================================================================

if __name__ == "__main__":
    ds = SyntheticMTLDataset(n_samples=100)
    sample = ds[0]
    print(f"Input shape: {sample['input'].shape}")
    for name, val in sample['targets'].items():
        print(f"  {name}: {val.item():.4f}")
    print(f"\nDataset size: {len(ds)}")
    print(f"Tasks: {ds.task_names}")

    # Compute expected loss scales
    import torch.nn.functional as F
    for name in ds.task_names:
        vals = ds.targets[name]
        cfg = next(c for c in ds.task_configs if c["name"] == name)
        if cfg["type"] == "mse":
            expected_loss = (vals ** 2).mean().item()
            print(f"  {name} (MSE): expected_loss ~ {expected_loss:.1f}")
        else:
            expected_loss = F.binary_cross_entropy(torch.ones_like(vals) * 0.5, vals).item()
            print(f"  {name} (BCE): expected_loss ~ {expected_loss:.4f}")
