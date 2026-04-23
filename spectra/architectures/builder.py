"""
spectra/architectures/builder.py
--------------------------------
Centralized Neural Architecture Factory.

Extracts the monolithic model building logic, providing uniform
access to the SharedTrunk, ALB, and specific Heads for all domains.
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
            
            # Robust extraction from global or nested config
            self.use_alb = cfg.get("use_alb") or cfg.get("method", {}).get("use_alb", False)
            if self.use_alb:
                backbone_name = cfg.get("backbone") or cfg.get("model", {}).get("backbone", "shared_trunk")
                if backbone_name == "segnet":
                    from spectra.core.alb import SpatialALB
                    self.alb = SpatialALB(
                        encoder=self.backbone,
                        d_model=cfg.model.d_model,
                    )
                else:
                    from spectra.core.alb import AsymmetricLatentBottleneck
                    self.alb = AsymmetricLatentBottleneck(
                        encoder=self.backbone,
                        input_dim=cfg.get("input_dim") or cfg.get("model", {}).get("input_dim"),
                        d_model=cfg.get("d_model") or cfg.get("model", {}).get("d_model"),
                        n_expert_layers=cfg.get("n_expert_layers") or cfg.get("method", {}).get("n_expert_layers", 3),
                        n_heads=cfg.get("n_heads") or cfg.get("model", {}).get("n_heads", 8),
                        dropout=cfg.get("dropout") or cfg.get("model", {}).get("dropout", 0.1),
                        init_mode=cfg.get("expert_init") or cfg.get("method", {}).get("expert_init", "orthogonal"),
                    )
            else:
                self.alb = None

            self.heads = nn.ModuleDict()
            # Store per-task manifold assignments for ALB routing
            self._task_manifolds = {}
            for task_cfg in cfg.tasks:
                d_head = cfg.get("d_model") or cfg.get("model", {}).get("d_model")
                manifold = task_cfg.get("manifold", "planner")
                self._task_manifolds[task_cfg.name] = manifold
                if self.use_alb and manifold == "both":
                    d_head = (cfg.get("d_model") or cfg.get("model", {}).get("d_model")) * 2
                self.heads[task_cfg.name] = build_head(task_cfg, d_head)

        def forward(self, x):
            if self.alb is not None:
                # ALB internally wraps the backbone (self.encoder = backbone).
                # Raw input goes directly to ALB, which runs backbone internally,
                # then produces the decoupled planner + expert manifolds.
                features = self.alb(x)

                # Manifold-Aware Head Routing (Core of Spectral Decoupling)
                # Each task head receives features from its designated manifold:
                #   - "planner": smooth, low-freq features (generative tasks like MSE)
                #   - "expert":  sharp, high-freq features (discriminative tasks like BCE)
                #   - "both":    concatenated [planner || expert] (hybrid tasks)
                f_planner = features.get("planner")
                f_expert = features.get("expert")
                if f_planner is None:
                    f_planner = next(iter(features.values()))
                if f_expert is None:
                    f_expert = f_planner

                outputs = {}
                for name, head in self.heads.items():
                    manifold = self._task_manifolds.get(name, "planner")
                    if manifold == "expert":
                        outputs[name] = head(f_expert)
                    elif manifold == "both":
                        concat_dim = 1 if f_planner.dim() == 4 else -1
                        outputs[name] = head(torch.cat([f_planner, f_expert], dim=concat_dim))
                    else:  # "planner" (default)
                        outputs[name] = head(f_planner)
                return outputs
            else:
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
