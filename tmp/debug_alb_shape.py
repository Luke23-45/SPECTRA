import sys
import os
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectra.architectures.builder import build_model
from spectra.data.synthetic import SyntheticMTLDataset

# Mock config mimicking bpgs_alb yaml merge
cfg = OmegaConf.create({
    "model": {
        "backbone": "shared_trunk",
        "input_dim": 20,
        "d_model": 128,
        "hidden_layers": 3,
        "n_heads": 8,
        "dropout": 0.1
    },
    "use_alb": True,
    "method": {
        "use_alb": True,
        "n_expert_layers": 3,
        "expert_init": "orthogonal"
    },
    "tasks": [
        {"name": "mse_high", "type": "regression", "manifold": "planner", "output_dim": 1},
        {"name": "bce_0", "type": "classification", "manifold": "expert", "num_classes": 1}
    ]
})

model = build_model(cfg)

ds = SyntheticMTLDataset(n_samples=5)
batch = [ds[i] for i in range(4)]
collated = SyntheticMTLDataset.collate_fn(batch)
x = collated["input"]

print(f"collated x shape: {x.shape}")
outputs = model(x)
p_ctx = model.alb.last_planner_ctx_seq
print(f"p_ctx shape: {p_ctx.shape}")
