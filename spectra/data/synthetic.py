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
    ):
        super().__init__()
        self.task_configs = task_configs or self.DEFAULT_TASKS

        gen = torch.Generator().manual_seed(seed)

        # Generate shared features
        self.X = torch.randn(n_samples, input_dim, generator=gen)
        W_shared = torch.randn(input_dim, hidden_dim, generator=gen) * 0.1
        Z = self.X @ W_shared  # [N, hidden_dim]

        # Generate per-task targets
        self.targets = {}
        for cfg in self.task_configs:
            W_task = torch.randn(hidden_dim, 1, generator=gen) * cfg["scale"]

            if cfg["type"] == "mse":
                y = Z @ W_task + cfg["offset"]  # [N, 1]
                self.targets[cfg["name"]] = y.squeeze(-1)

            elif cfg["type"] == "bce":
                logits = Z @ W_task
                y = (logits > 0).float().squeeze(-1)  # [N]
                self.targets[cfg["name"]] = y

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
