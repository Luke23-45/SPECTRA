"""
spectra/engine/trainer.py
-------------------------
Main LightningModule wrapper for SPECTRA experiments.

Supports all combinations of:
- Backbones: SharedTrunk (synthetic/clinical), SegNet/MTAN (NYUv2, future)
- Weighters: B-PGS, Kendall, UW-SO, PCGrad, NTKMTL, Static
- ALB: Enabled or disabled (ablation toggle)

This is the central orchestration point that connects backbone → ALB → heads → weighter.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from spectra.core.bpgs import BPGSScaler
from spectra.core.alb import AsymmetricLatentBottleneck
from spectra.baselines import build_weighter
from spectra.baselines.pcgrad import PCGradWeighter
from spectra.backbones.shared_trunk import SharedTrunk
from spectra.heads.task_heads import RegressionHead, ClassificationHead
from spectra.engine.schedulers import get_cosine_schedule_with_warmup
from spectra.evaluation.metrics import SegmentationMetrics, DepthMetrics, NormalMetrics

logger = logging.getLogger("spectra.trainer")


def _build_backbone(cfg: DictConfig) -> nn.Module:
    """Factory: builds backbone from config."""
    name = cfg.model.get("backbone", "shared_trunk")
    if name == "shared_trunk":
        return SharedTrunk(
            input_dim=cfg.model.input_dim,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.get("hidden_layers", 3),
            dropout=cfg.model.get("dropout", 0.1),
        )
    elif name == "segnet":
        from spectra.backbones.segnet import SegNet
        return SegNet(
            input_channels=cfg.model.get("input_channels", 3),
            d_model=cfg.model.d_model,
        )
    else:
        raise ValueError(f"Unknown backbone: {name}. Available: shared_trunk, segnet")


def _build_head(task_cfg: DictConfig, d_model: int) -> nn.Module:
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


def _build_loss(task_cfg: DictConfig):
    """Factory: builds loss function from config."""
    loss_name = task_cfg.get("loss", "mse")
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "cross_entropy":
        ignore_idx = task_cfg.get("ignore_index", -100)
        return nn.CrossEntropyLoss(ignore_index=ignore_idx)
    elif loss_name == "cosine":
        return nn.CosineEmbeddingLoss()
    elif loss_name == "masked_l1":
        from spectra.engine.dense_losses import MaskedL1Loss
        return MaskedL1Loss()
    elif loss_name == "cosine_dense":
        from spectra.engine.dense_losses import DenseCosineLoss
        return DenseCosineLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


class SPECTRAModule(pl.LightningModule):
    """
    Unified training wrapper for SPECTRA experiments.

    Handles the complete forward → loss → weight → backward pipeline
    for any combination of backbone, ALB, heads, and weighting method.

    Args:
        cfg: Full Hydra DictConfig.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # save_hyperparameters needs a dict, not DictConfig directly
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg

        # ─── 1. Build Backbone ────────────────────────────────────
        self.backbone = _build_backbone(cfg)

        # ─── 2. Build ALB (optional) ─────────────────────────────
        self.use_alb = cfg.method.get("use_alb", False)
        if self.use_alb:
            backbone_name = cfg.model.get("backbone", "shared_trunk")
            
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
                    input_dim=cfg.model.get("input_dim"), # May be None for some models
                    d_model=cfg.model.d_model,
                    n_expert_layers=cfg.method.get("n_expert_layers", 3),
                    n_heads=cfg.model.get("n_heads", 8),
                    dropout=cfg.model.get("dropout", 0.1),
                    init_mode=cfg.method.get("expert_init", "orthogonal"),
                )

        # ─── 3. Build Task Heads ────────────────────────────────
        self.heads = nn.ModuleDict()
        self.task_losses = nn.ModuleDict()
        self.task_types = {}
        self.task_manifolds = {}

        for task_cfg in cfg.tasks:
            name = task_cfg.name
            d_head = cfg.model.d_model
            # If using ALB with "both" manifold, heads receive 2×d_model
            if self.use_alb and task_cfg.get("manifold", "planner") == "both":
                d_head = cfg.model.d_model * 2

            self.heads[name] = _build_head(task_cfg, d_head)
            self.task_losses[name] = _build_loss(task_cfg)
            self.task_types[name] = task_cfg.get("type", "regression")
            self.task_manifolds[name] = task_cfg.get("manifold", "planner")

        self.task_names = [t.name for t in cfg.tasks]
        self.num_tasks = len(self.task_names)

        # ─── 4. Build Weighter ─────────────────────────────────
        self.weighter = build_weighter(cfg)
        self.is_pcgrad = isinstance(self.weighter, PCGradWeighter)

        # PCGrad needs manual optimization because it computes its own
        # per-task gradients via autograd.grad — PL automatic backward
        # would conflict with this by calling .backward() on the sum
        # before PCGrad has a chance to project gradients.
        if self.is_pcgrad:
            self.automatic_optimization = False

        logger.info(
            f"[SPECTRA] Initialized: backbone={cfg.model.get('backbone', 'shared_trunk')}, "
            f"method={cfg.method.name}, ALB={self.use_alb}, tasks={self.num_tasks}"
        )

        # ─── 5. Initialize Task Metrics ─────────────────────────
        # Metrics are initialized here so they are always available,
        # even before the first validation epoch.
        self._init_metrics()

    def _init_metrics(self) -> None:
        """
        Initialize per-task evaluation metric accumulators.

        Detects which tasks are present in the config and creates the
        appropriate metric object. Non-NYUv2 tasks get no metric object
        (losses suffice for synthetic/clinical stop-go decisions).
        """
        self._val_metrics: Dict[str, Any] = {}
        num_classes = 13
        ignore_idx  = 255

        for task_cfg in self.cfg.tasks:
            name = task_cfg.name
            t    = task_cfg.get("type", "regression")

            if name == "segmentation" or t == "dense_classification":
                num_classes = task_cfg.get("num_classes", 13)
                ignore_idx  = task_cfg.get("ignore_index", 255)
                self._val_metrics[name] = SegmentationMetrics(
                    num_classes=num_classes, ignore_index=ignore_idx
                )
            elif name == "depth" or (t == "dense_regression" and task_cfg.get("output_dim", 1) == 1
                                     and "depth" in name):
                self._val_metrics[name] = DepthMetrics(max_depth=10.0)
            elif name == "normals" or (t == "dense_regression" and "normal" in name):
                self._val_metrics[name] = NormalMetrics()
            # Scalar regression/classification tasks: loss serves as proxy metric.

        if self._val_metrics:
            logger.info(
                f"[SPECTRA] Task metrics initialized: {list(self._val_metrics.keys())}"
            )

    # =================================================================
    # FORWARD
    # =================================================================

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input"]

        # ALB requires [B, T, C_in]. If input is [B, C_in] (no temporal dim),
        # add a dummy temporal dimension of 1 so inception convolutions work.
        needs_unsqueeze = False
        if self.use_alb and x.dim() == 2:
            x = x.unsqueeze(1)  # [B, C] → [B, 1, C]
            needs_unsqueeze = True

        if self.use_alb:
            alb_out = self.alb(x)
            predictions = {}
            for name, head in self.heads.items():
                manifold = self.task_manifolds[name]
                task_type = self.task_types[name]
                
                # Agnostic Dispatch: 
                # - Dense tasks (Vision) use the full lattice (planner/expert)
                # - Scalar tasks (Clinical/Synthetic) use global summaries (planner_global/expert_global)
                is_dense = task_type.startswith("dense")
                
                if manifold == "both":
                    ctx_planner = alb_out["planner"] if is_dense else alb_out["planner_global"]
                    ctx_expert = alb_out["expert"] if is_dense else alb_out["expert_global"]
                    # Cat on channel dim (1 for 4D, -1 for 3D/2D)
                    dim = 1 if ctx_planner.dim() == 4 else -1
                    ctx = torch.cat([ctx_planner, ctx_expert], dim=dim)
                else:
                    key = manifold if is_dense else f"{manifold}_global"
                    ctx = alb_out[key]
                
                predictions[name] = head(ctx)
        else:
            features = self.backbone(x)
            predictions = {}
            for name, head in self.heads.items():
                predictions[name] = head(features)

        return predictions

    # =================================================================
    # TRAINING STEP
    # =================================================================

    def training_step(self, batch: Dict, batch_idx: int) -> Optional[torch.Tensor]:
        predictions = self(batch)

        # Compute per-task losses
        task_loss_list = []
        loss_dict = {}

        for name in self.task_names:
            pred = predictions[name]
            target = batch["targets"][name]

            # Shape alignment (for scalar tasks only — dense tasks have matched shapes)
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim():
                    pred = pred.squeeze(-1)
                if target.dim() > pred.dim():
                    target = target.squeeze(-1)

            # Masked losses (depth, normals) need access to metadata
            loss_fn = self.task_losses[name]
            if hasattr(loss_fn, 'requires_mask') and loss_fn.requires_mask and "meta" in batch:
                loss = loss_fn(pred, target, batch["meta"])
            else:
                loss = loss_fn(pred, target)
            task_loss_list.append(loss)
            loss_dict[name] = loss

        losses_tensor = torch.stack(task_loss_list)

        # ─── PCGrad: Manual optimization path ─────────────────
        if self.is_pcgrad:
            opt = self.optimizers()
            sch = self.lr_schedulers()

            opt.zero_grad()

            # Step 1: PCGrad projects gradients onto shared params
            # When ALB is enabled, backbone is wrapped inside alb.encoder.
            # We need to pass backbone params (the actual shared trunk).
            shared_params_for_pcgrad = list(self.backbone.parameters())
            pcgrad_metrics = self.weighter.backward_and_project(
                task_loss_list,
                shared_params_for_pcgrad,
            )

            # Step 2: Save the projected backbone grads (PCGrad spent O(N²P) computing these)
            saved_backbone_grads = {
                id(p): p.grad.clone() for p in self.backbone.parameters()
                if p.grad is not None
            }

            # Step 3: Backward through heads to get gradients for head params
            # This WILL overwrite backbone grads — that's why we saved them above
            head_loss = losses_tensor.sum()
            self.manual_backward(head_loss)

            # Step 4: Restore projected backbone grads (overwriting the naive ones)
            for p in self.backbone.parameters():
                pid = id(p)
                if pid in saved_backbone_grads:
                    p.grad = saved_backbone_grads[pid]

            if self.cfg.train.get("grad_clip", 0) > 0:
                self.clip_gradients(opt, gradient_clip_val=self.cfg.train.grad_clip)
            opt.step()
            if sch is not None:
                sch.step()

            total_loss = losses_tensor.sum().detach()
            w_metrics = pcgrad_metrics
        else:
            # ─── Standard optimization path ───────────────────
            shared_params = list(self.backbone.parameters())
            total_loss, w_metrics = self.weighter(
                losses_tensor,
                shared_params=shared_params,
                sync_ddp=self.trainer.world_size > 1 if self.trainer else False,
            )

        # NaN guard
        if torch.isnan(total_loss):
            logger.warning(f"[Step {self.global_step}] NaN loss detected — skipping batch")
            if self.is_pcgrad:
                return None
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True, sync_dist=True)
        for name, loss in loss_dict.items():
            self.log(f"train/{name}_loss", loss, sync_dist=True)
        for key, val in w_metrics.items():
            self.log(f"train/{key}", val, sync_dist=True)

        # For PCGrad, return None (manual optimization).
        # For others, return total_loss for PL automatic backward.
        return None if self.is_pcgrad else total_loss

    # =================================================================
    # VALIDATION STEP
    # =================================================================

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch)

        total_val_loss = torch.tensor(0.0, device=self.device)
        for name in self.task_names:
            pred   = predictions[name]
            target = batch["targets"][name]

            # Shape alignment (for scalar tasks only)
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim():
                    pred = pred.squeeze(-1)
                if target.dim() > pred.dim():
                    target = target.squeeze(-1)

            # Masked losses need metadata
            loss_fn = self.task_losses[name]
            if hasattr(loss_fn, 'requires_mask') and loss_fn.requires_mask and "meta" in batch:
                loss = loss_fn(pred, target, batch["meta"])
            else:
                loss = loss_fn(pred, target)
            self.log(f"val/{name}_loss", loss, sync_dist=True, prog_bar=False)
            total_val_loss = total_val_loss + loss

            # ── Accumulate task metrics (no compute yet — done at epoch end) ──
            if name in self._val_metrics:
                metric_obj = self._val_metrics[name]
                if isinstance(metric_obj, SegmentationMetrics):
                    metric_obj.update(pred, target)
                elif isinstance(metric_obj, DepthMetrics):
                    # Pass depth_mask from meta if available
                    mask = batch.get("meta", {}).get("depth_mask", None)
                    metric_obj.update(pred, target, mask=mask)
                elif isinstance(metric_obj, NormalMetrics):
                    metric_obj.update(pred, target)

        self.log("val/total_loss", total_val_loss, sync_dist=True, prog_bar=True)

    # =================================================================
    # VALIDATION EPOCH HOOKS
    # =================================================================

    def on_validation_epoch_start(self) -> None:
        """Reset all metric accumulators at the start of each val epoch."""
        for m in self._val_metrics.values():
            m.reset()

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate accumulated per-batch metrics and log to W&B / console.

        This is the only place where final metric values are computed.
        Runs on rank 0 only (metrics are CPU-side and not distributed).
        """
        if not self._val_metrics:
            return

        # ── Segmentation: mIoU ──────────────────────────────────────
        for name, metric_obj in self._val_metrics.items():
            if isinstance(metric_obj, SegmentationMetrics):
                r = metric_obj.compute()
                self.log(f"val/{name}_miou",           r["miou"],           prog_bar=True,  sync_dist=False)
                self.log(f"val/{name}_pixel_acc",      r["pixel_acc"],      prog_bar=False, sync_dist=False)
                self.log(f"val/{name}_mean_class_acc", r["mean_class_acc"], prog_bar=False, sync_dist=False)
                # Convenience alias for ModelCheckpoint monitor
                if name == "segmentation":
                    self.log("val/miou", r["miou"], prog_bar=True, sync_dist=False)
                logger.info(
                    f"[Val] {name}: mIoU={r['miou']:.4f}, "
                    f"PixAcc={r['pixel_acc']:.4f}, "
                    f"N_classes={r['n_valid_classes']}"
                )

            elif isinstance(metric_obj, DepthMetrics):
                r = metric_obj.compute()
                self.log(f"val/{name}_abs_rel",  r["abs_rel"],  prog_bar=True,  sync_dist=False)
                self.log(f"val/{name}_rmse",      r["rmse"],     prog_bar=False, sync_dist=False)
                self.log(f"val/{name}_delta_1",   r["delta_1"],  prog_bar=False, sync_dist=False)
                self.log(f"val/{name}_log_rmse",  r["log_rmse"], prog_bar=False, sync_dist=False)
                if name == "depth":
                    self.log("val/depth_abs_rel", r["abs_rel"], prog_bar=True, sync_dist=False)
                logger.info(
                    f"[Val] {name}: abs_rel={r['abs_rel']:.4f}, "
                    f"RMSE={r['rmse']:.4f}, δ<1.25={r['delta_1']:.4f}"
                )

            elif isinstance(metric_obj, NormalMetrics):
                r = metric_obj.compute()
                self.log(f"val/{name}_mean_angle",   r["mean_angle_deg"],   prog_bar=True,  sync_dist=False)
                self.log(f"val/{name}_median_angle",  r["median_angle_deg"], prog_bar=False, sync_dist=False)
                self.log(f"val/{name}_within_11_25",  r["within_11_25"],     prog_bar=False, sync_dist=False)
                if name == "normals":
                    self.log("val/normals_mean_angle", r["mean_angle_deg"], prog_bar=True, sync_dist=False)
                logger.info(
                    f"[Val] {name}: mean_angle={r['mean_angle_deg']:.2f}°, "
                    f"<11.25°={r['within_11_25']:.4f}"
                )

        # ── B-PGS Telemetry at Epoch End ────────────────────────────
        if hasattr(self.weighter, "get_telemetry"):
            tel = self.weighter.get_telemetry()
            for i, lv in enumerate(tel.get("log_vars", [])):
                self.log(f"val/bpgs_log_var_{i}", lv, sync_dist=False)


    # =================================================================
    # OPTIMIZER & SCHEDULER
    # =================================================================

    def configure_optimizers(self):
        # Separate weight-decay params from norms/biases
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # CRITICAL: Weighter parameters MUST NOT get weight decay, otherwise
            # log_vars/theta will be regularized towards 0, destroying Bayesian learning!
            no_weight_decay_keywords = [
                "bias", "norm", "bn", "LayerNorm",
                "weighter", "log_vars", "theta"
            ]
            if any(nd in name for nd in no_weight_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.cfg.train.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.cfg.train.lr)

        # Cosine warmup scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            min_lr_ratio=self.cfg.train.get("min_lr", 1e-6) / self.cfg.train.lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
