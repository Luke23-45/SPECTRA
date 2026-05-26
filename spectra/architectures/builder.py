"""
Centralized neural architecture factory.

Provides uniform access to the SharedTrunk and task-specific heads
for all domains.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from spectra.backbones.shared_trunk import SharedTrunk
from spectra.heads.task_heads import RegressionHead, ClassificationHead

def build_model(cfg: DictConfig) -> nn.Module:
    """Wrapper entry point. Backwards compatibility for the model extraction."""
    class DynamicWrapper(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.backbone = build_backbone(cfg)

            self.heads = nn.ModuleDict()
            for task_cfg in cfg.tasks:
                d_head = cfg.get("d_model") or cfg.get("model", {}).get("d_model")
                self.heads[task_cfg.name] = build_head(task_cfg, d_head)

        def forward(self, x):
            features = self.backbone(x)
            outputs = {}
            for name, head in self.heads.items():
                outputs[name] = head(features)
            return outputs

    return DynamicWrapper(cfg)


def build_backbone(cfg: DictConfig) -> nn.Module:
    """Factory: builds backbone from config."""
    name = cfg.get("backbone") or cfg.get("model", {}).get("backbone", "shared_trunk")
    if name == "shared_trunk":
        # Robustly extract SharedTrunk parameters
        d_model = cfg.get("d_model") or cfg.get("model", {}).get("d_model")
        input_dim = cfg.get("input_dim") or cfg.get("model", {}).get("input_dim")
        n_layers = cfg.get("hidden_layers") or cfg.get("model", {}).get("hidden_layers", 3)
        dropout = cfg.get("dropout") or cfg.get("model", {}).get("dropout", 0.1)
        
        return SharedTrunk(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
    elif name == "segnet":
        from spectra.backbones.segnet import SegNet
        return SegNet(
            input_channels=cfg.model.get("input_channels", 3),
            d_model=cfg.model.d_model,
        )
    else:
        raise ValueError(f"Unknown backbone: {name}. Available: shared_trunk, segnet")


def build_head(task_cfg: DictConfig, d_model: int) -> nn.Module:
    """Factory: builds task head from config."""
    task_type = task_cfg.get("type", "regression")
    if task_type == "dense_regression":
        from spectra.heads.dense_heads import DenseRegressionHead
        return DenseRegressionHead(d_model, output_dim=task_cfg.get("output_dim", 1))
    elif task_type == "dense_classification":
        from spectra.heads.dense_heads import DenseSegmentationHead
        return DenseSegmentationHead(d_model, num_classes=task_cfg.get("num_classes", 13))
    elif task_type == "regression":
        return RegressionHead(d_model, output_dim=task_cfg.get("output_dim", 1))
    elif task_type == "classification":
        return ClassificationHead(d_model, num_classes=task_cfg.get("num_classes", 1))
    else:
        raise ValueError(f"Unknown task type: {task_type}")
