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
import torch.distributed as dist

from spectra.core.bpgs import BPGS
from spectra.core.alb import AsymmetricLatentBottleneck
from spectra.baselines import build_weighter
from spectra.baselines.pcgrad import PCGradWeighter
from spectra.backbones.shared_trunk import SharedTrunk
from spectra.heads.task_heads import RegressionHead, ClassificationHead
from spectra.engine.schedulers import get_cosine_schedule_with_warmup
from spectra.evaluation.metrics import SegmentationMetrics, DepthMetrics, NormalMetrics

try:
    from torchmetrics import MetricCollection
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryRecall, MulticlassAUROC, MulticlassAccuracy
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

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
        pos_weight = task_cfg.get("pos_weight", None)
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
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


def _build_val_loss(task_cfg: DictConfig):
    """Factory: builds UNWEIGHTED loss for validation (no pos_weight bias).

    Why separate? Training uses pos_weight to compensate for sepsis_boost
    oversampling, but validation runs on the natural distribution. Using
    pos_weight in val inflates losses on the rare positives and creates a
    misleading loss signal that diverges from true model quality.
    """
    loss_name = task_cfg.get("loss", "mse")
    if loss_name == "bce":
        # Deliberately omit pos_weight for unbiased validation
        return nn.BCEWithLogitsLoss()
    elif loss_name == "cross_entropy":
        ignore_idx = task_cfg.get("ignore_index", -100)
        return nn.CrossEntropyLoss(ignore_index=ignore_idx)
    else:
        # For all other loss types, val loss == train loss
        return _build_loss(task_cfg)



class SPECTRAModule(pl.LightningModule):
    """
    Unified training wrapper for SPECTRA experiments.

    Handles the complete forward → loss → weight → backward pipeline
    for any combination of backbone, ALB, heads, and weighting method.

    Args:
        cfg: Full Hydra DictConfig.
    """

    def __init__(self, cfg: DictConfig, engine: Any = None):
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
        self._val_losses = nn.ModuleDict()  # Unbiased val losses (no pos_weight)
        self.task_types = {}
        self.task_manifolds = {}
        self.task_weights = {}

        for task_cfg in cfg.tasks:
            name = task_cfg.name
            d_head = cfg.model.d_model
            # If using ALB with "both" manifold, heads receive 2×d_model
            if self.use_alb and task_cfg.get("manifold", "planner") == "both":
                d_head = cfg.model.d_model * 2

            self.heads[name] = _build_head(task_cfg, d_head)
            self.task_losses[name] = _build_loss(task_cfg)
            self._val_losses[name] = _build_val_loss(task_cfg)
            self.task_types[name] = task_cfg.get("type", "regression")
            self.task_manifolds[name] = task_cfg.get("manifold", "planner")
            self.task_weights[name] = float(task_cfg.get("weight", 1.0))

        self.task_names = [t.name for t in cfg.tasks]
        self.num_tasks = len(self.task_names)

        # ─── 4. Build Weighter ─────────────────────────────────
        self.weighter = build_weighter(cfg)
        self.is_pcgrad = isinstance(self.weighter, PCGradWeighter)
        self.is_bpgs = isinstance(self.weighter, BPGS)

        # [SOTA Fix] BPGS and PCGrad both require manual optimization.
        # BPGS requires decoupled gradient flows (Network vs Uncertainty).
        # PCGrad requires per-task gradient projections.
        if self.is_pcgrad or self.is_bpgs:
            self.automatic_optimization = False

        logger.info(
            f"[SPECTRA] Initialized: backbone={cfg.model.get('backbone', 'shared_trunk')}, "
            f"method={cfg.method.name}, ALB={self.use_alb}, tasks={self.num_tasks}"
        )

        # ─── 5. Initialize Task Metrics ─────────────────────────
        # Metrics are initialized here so they are always available,
        # even before the first validation epoch.
        self._init_metrics()
        
        # ─── 6. Live Training Metrics (SOTA Bar) ───────────────
        self._init_train_metrics()

    def _init_train_metrics(self) -> None:
        """Initialize real-time metrics for the training progress bar.
        NOTE: Heavy rank-based metrics (AUC, PRC) are disabled here to prevent
        O(N^2) accumulation slowdowns during the training epoch. They are correctly
        computed per-epoch during validation.
        """
        self.train_metrics = nn.ModuleDict()
        # Removed BinaryAUROC and BinaryAveragePrecision from training step
        # to guarantee 14+ it/s throughput.

    def _init_metrics(self) -> None:
        """
        Initialize per-task evaluation metric accumulators.

        Detects which tasks are present in the config and creates the
        appropriate metric object. Non-NYUv2 tasks get no metric object
        (losses suffice for synthetic/clinical stop-go decisions).
        """
        self._val_metrics = nn.ModuleDict()
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
            elif name == "outcome" and t == "classification":
                # [SOTA Fix] Restoring Sepsis classification metrics to Validation!
                self._val_metrics[name] = MetricCollection({
                    "AUC": BinaryAUROC(),
                    "PRC": BinaryAveragePrecision(),
                    "R": BinaryRecall()
                })
            elif t == "classification" and task_cfg.get("num_classes", 1) > 1:
                nc = task_cfg["num_classes"]
                self._val_metrics[name] = MetricCollection({
                    "ACC": MulticlassAccuracy(num_classes=nc),
                    "AUC": MulticlassAUROC(num_classes=nc)
                })

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
        weighted_task_loss_list = []
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

            # [SOTA Fix] Label smoothing for binary classification to prevent
            # overconfidence on the boosted training distribution.
            # Smooths targets: 1.0 → 0.975, 0.0 → 0.025
            smooth_target = target
            if name == "outcome" and self.task_types.get(name) == "classification":
                eps = 0.05
                smooth_target = target * (1.0 - eps) + 0.5 * eps

            # Masked losses (depth, normals) need access to metadata
            loss_fn = self.task_losses[name]
            if hasattr(loss_fn, 'requires_mask') and loss_fn.requires_mask and "meta" in batch:
                loss = loss_fn(pred, target, batch["meta"])
            else:
                loss = loss_fn(pred, smooth_target)
            task_loss_list.append(loss)
            weighted_task_loss_list.append(loss * self.task_weights[name])
            loss_dict[name] = loss

        losses_tensor = torch.stack(weighted_task_loss_list)

        # ─── PCGrad SOTA: Manual Optimization & Surgery ────────────────
        if self.is_pcgrad:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            opt.zero_grad()

            scaler = getattr(self.trainer.precision_plugin, "scaler", None)
            raw_opt = opt.optimizer if hasattr(opt, "optimizer") else opt
            shared_params = list(self.backbone.parameters())
            if self.use_alb:
                shared_params += list(self.alb.parameters())
            bsz = batch.get("input").shape[0]

            if scaler is not None:
                # ═══════════════════════════════════════════════════════════
                # SOTA AMP-Safe PCGrad Pipeline (v2 — Clean Gradient Flow)
                #
                # Key insight: GradScaler's scale factor is a uniform scalar,
                # so PCGrad's dot-product comparisons are scale-invariant.
                # We compute ALL gradients via autograd.grad on SCALED losses,
                # do surgery on the scaled gradients, then let scaler.unscale_()
                # uniformly remove the scaling. This avoids the magnitude
                # mismatch bug where backbone got raw fp16 gradients while
                # heads got properly scaled/unscaled gradients.
                # ═══════════════════════════════════════════════════════════

                # For NaN guard and logging (no gradient involvement)
                total_loss = losses_tensor.sum()

                # 1. Per-task backbone gradients (SCALED magnitude)
                task_grads = []
                for i, task_loss in enumerate(weighted_task_loss_list):
                    grads = torch.autograd.grad(
                        scaler.scale(task_loss), shared_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    grads = [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(grads, shared_params)
                    ]
                    task_grads.append(grads)

                # 2. PCGrad surgery + assignment to backbone .grad
                pcgrad_metrics = self.weighter.project_and_assign(
                    task_grads, shared_params
                )

                # 3. Head gradients: each head sees ONLY its own task loss
                for idx, (task_name, task_loss) in enumerate(zip(self.task_names, weighted_task_loss_list)):
                    head_params = list(self.heads[task_name].parameters())
                    if not head_params:
                        continue
                    
                    is_last_head = (idx == self.num_tasks - 1)
                    head_grads = torch.autograd.grad(
                        scaler.scale(task_loss), head_params,
                        retain_graph=not is_last_head,  # Free graph on the last head
                        allow_unused=True,
                    )
                    for p, g in zip(head_params, head_grads):
                        if g is not None:
                            p.grad = g

                # 4. Unscale ALL gradients (backbone + heads) uniformly
                scaler.unscale_(raw_opt)

                # [Patch 3: PCGrad DDP Sync]
                # Synchronize ALL gradients across GPUs. Both backbone and heads must be synced.
                if self.trainer.world_size > 1 and dist.is_initialized():
                    for p in self.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                # 5. Gradient clipping (on properly-unscaled gradients)
                if self.cfg.train.get("grad_clip", 0) > 0:
                    self.clip_gradients(opt, gradient_clip_val=self.cfg.train.grad_clip)

                # 6. NaN-safe Optimizer Step
                old_scale = scaler.get_scale()
                scaler.step(raw_opt)
                scaler.update()

                # 7. Scheduler Sync — only step if optimizer actually updated
                if sch is not None and scaler.get_scale() >= old_scale:
                    sch.step()
            else:
                # ═══════════════════════════════════════════════════════════
                # Standard FP32 Pipeline (same clean gradient flow)
                # ═══════════════════════════════════════════════════════════

                total_loss = losses_tensor.sum()

                # 1. Per-task backbone gradients
                task_grads = []
                for i, task_loss in enumerate(weighted_task_loss_list):
                    grads = torch.autograd.grad(
                        task_loss, shared_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    grads = [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(grads, shared_params)
                    ]
                    task_grads.append(grads)

                # 2. PCGrad surgery + assignment
                pcgrad_metrics = self.weighter.project_and_assign(
                    task_grads, shared_params
                )

                # 3. Head gradients (per-task isolation)
                for idx, (task_name, task_loss) in enumerate(zip(self.task_names, weighted_task_loss_list)):
                    head_params = list(self.heads[task_name].parameters())
                    if not head_params:
                        continue
                    
                    is_last_head = (idx == self.num_tasks - 1)
                    head_grads = torch.autograd.grad(
                        task_loss, head_params,
                        retain_graph=not is_last_head,  # Free graph on the last head
                        allow_unused=True,
                    )
                    for p, g in zip(head_params, head_grads):
                        if g is not None:
                            p.grad = g

                # [Patch 3: PCGrad DDP Sync]
                if self.trainer.world_size > 1 and dist.is_initialized():
                    for p in self.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                # 4. Gradient clipping
                if self.cfg.train.get("grad_clip", 0) > 0:
                    self.clip_gradients(opt, gradient_clip_val=self.cfg.train.grad_clip)

                # 5. Step
                opt.step()
                if sch is not None:
                    sch.step()

            # --- Telemetry & Health ---
            total_loss_val = losses_tensor.sum().detach()
            gn = torch.norm(torch.stack([p.grad.detach().norm(2) for p in shared_params if p.grad is not None]), 2)
            
            self.log("health/backbone_grad_norm", gn, prog_bar=True, on_step=True, batch_size=bsz)
            self.log("pcgrad/total_conflicts", pcgrad_metrics.get("pcgrad/total_conflicts", 0), prog_bar=True, on_step=True, batch_size=bsz)
            w_metrics = pcgrad_metrics
        elif self.is_bpgs:
            # ─── B-PGS SOTA: Decoupled Manual Optimization ────────────
            # Reference: BPGS Final Synthesized Definition
            opts = self.optimizers()
            opt_net = opts[0]
            opt_unc = opts[1]
            scaler = getattr(self.trainer.precision_plugin, "scaler", None)
            
            # 1. Update smoothed losses (Signal integration via NaN-gated EMA)
            # MUST be called before loss computation.
            self.weighter.update_ema(weighted_task_loss_list)
            
            # 2. Base Flow Step (Network weights w)
            # Backprop only through precision-weighted task losses.
            opt_net.zero_grad()
            loss_net = self.weighter.network_loss(weighted_task_loss_list)
            
            if scaler is not None:
                self.manual_backward(scaler.scale(loss_net))
                scaler.unscale_(opt_net)
            else:
                self.manual_backward(loss_net)
            
            if self.cfg.train.get("grad_clip", 0) > 0:
                self.clip_gradients(opt_net, gradient_clip_val=self.cfg.train.grad_clip)
                
            if scaler is not None:
                scaler.step(opt_net)
            else:
                opt_net.step()
                
            # 3. Fiber Flow Step (Uncertainty parameters theta)
            # Backprop only through the uncertainty regularization manifold.
            opt_unc.zero_grad()
            loss_unc = self.weighter.uncertainty_loss()
            
            if scaler is not None:
                self.manual_backward(scaler.scale(loss_unc))
                scaler.unscale_(opt_unc)
            else:
                self.manual_backward(loss_unc)
                
            # [SOTA Requirement] Decoupled DDP Sync for Weighter Parameters
            if self.trainer.world_size > 1 and dist.is_initialized():
                for p in self.weighter.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            if scaler is not None:
                scaler.step(opt_unc)
                scaler.update()
            else:
                opt_unc.step()
            
            # 4. Schedule step (Standard schedule is tied to opt_net)
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()
                
            # Prepare for logging
            total_loss = loss_net
            w_metrics = self.weighter.get_task_stats()
        else:
            # ─── Standard optimization path ───────────────────
            shared_params = list(self.backbone.parameters())
            if self.use_alb:
                shared_params += list(self.alb.parameters())
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

        # Logging (Live updates for the standard TQDM bar)
        bsz = batch.get("input").shape[0]
        self.log("train/total_loss", total_loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bsz)
        for name, loss in loss_dict.items():
            self.log(f"train/{name}_loss", loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
            # (train_metrics updates removed due to O(N^2) complexity; AUC handled strictly in validation)

        for key, val in w_metrics.items():
            self.log(f"train/{key}", val, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.get("input").shape[0])

        # For PCGrad/BPGS, return None (manual optimization).
        # For others, return total_loss for PL automatic backward.
        return None if (self.is_pcgrad or self.is_bpgs) else total_loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        # [SPECTRA FIX] Post-step manifold projection for B-PGS
        # Reference: docs/models/wrapper_generalist.py L2803-2805
        # Prevents theta drift into sigmoid saturation zones
        if hasattr(self.weighter, 'project_parameters'):
            self.weighter.project_parameters()

        # Periodic health check: Log backbone weight norm to detect dying weights
        if batch_idx % 100 == 0:
            shared_params = list(self.backbone.parameters())
            if self.use_alb:
                shared_params += list(self.alb.parameters())
            wn = torch.norm(torch.stack([p.detach().norm(2) for p in shared_params]), 2)
            # Log as WN for the SOTA researcher
            self.log("health/backbone_weight_norm", wn, on_step=True, on_epoch=False, batch_size=batch.get("input").shape[0])

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

            # [SOTA Fix] Use UNBIASED val loss (no pos_weight) for true distribution
            val_loss_fn = self._val_losses[name]
            if hasattr(val_loss_fn, 'requires_mask') and val_loss_fn.requires_mask and "meta" in batch:
                loss = val_loss_fn(pred, target, batch["meta"])
            else:
                loss = val_loss_fn(pred, target)
            self.log(f"val/{name}_loss", loss, sync_dist=True, prog_bar=False, batch_size=batch.get("input").shape[0])
            total_val_loss = total_val_loss + loss * self.task_weights[name]

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
                elif isinstance(metric_obj, MetricCollection):
                    # [SOTA Fix] Accumulate classification metrics (AUC/PRC)
                    p = pred.squeeze(-1) if pred.dim() > 1 and pred.shape[-1] == 1 else pred
                    t = target.squeeze(-1) if target.dim() > 1 and target.shape[-1] == 1 else target

                    # Binary metrics expect int targets; multiclass metrics also
                    # require integer class indices. Keep this explicit for safety.
                    if name == "outcome":
                        # Convert logits to probabilities to fix `threshold=0.5` bug in BinaryRecall
                        metric_obj.update(torch.sigmoid(p), t.int())
                    else:
                        metric_obj.update(p, t.long())

        self.log("val/total_loss", total_val_loss, sync_dist=True, prog_bar=True, batch_size=batch.get("input").shape[0])

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
                    f"[Val] {name}: mean_angle={r['mean_angle_deg']:.2f} degrees, "
                    f"<11.25 degrees={r['within_11_25']:.4f}"
                )

            elif isinstance(metric_obj, MetricCollection):
                # [SOTA Fix] Compute global AUC/PRC for the epoch
                m_out = metric_obj.compute()
                for m_name, val in m_out.items():
                    self.log(f"val/{name}_{m_name}", val, prog_bar=True, sync_dist=False)
                
                # Global alias for the SOTA Bar
                if "AUC" in m_out:
                    self.log("val/AUC", m_out["AUC"], prog_bar=True, sync_dist=False)
                
                log_strs = [f"{k}={v:.4f}" for k, v in m_out.items()]
                logger.info(f"[Val] {name}: " + ", ".join(log_strs))

        # ── B-PGS Telemetry at Epoch End ────────────────────────────
        if hasattr(self.weighter, "get_telemetry"):
            tel = self.weighter.get_telemetry()
            for i, lv in enumerate(tel.get("log_vars", [])):
                self.log(f"val/bpgs_log_var_{i}", lv, sync_dist=False)


    # =================================================================
    # OPTIMIZER & SCHEDULER
    # =================================================================

    def configure_optimizers(self):
        # 1. Parameter Grouping: Separate Network from Uncertainty (for BPGS)
        # And separate Weight-Decay from Non-Decay (for all)
        net_decay = []
        net_no_decay = []
        unc_params = []

        no_decay_keywords = ["bias", "norm", "bn", "LayerNorm"]
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # If BPGS is active, its parameters MUST be in a separate optimizer
            if self.is_bpgs and "weighter" in name:
                unc_params.append(param)
            else:
                # Standard model parameters
                if any(nd in name for nd in no_decay_keywords):
                    net_no_decay.append(param)
                else:
                    net_decay.append(param)

        # 2. Build Optimizer(s)
        opt_net = torch.optim.AdamW([
            {"params": net_decay, "weight_decay": self.cfg.train.weight_decay},
            {"params": net_no_decay, "weight_decay": 0.0},
        ], lr=self.cfg.train.lr)

        # Cosine warmup scheduler for the main network
        sch_net = get_cosine_schedule_with_warmup(
            opt_net,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            min_lr_ratio=self.cfg.train.get("min_lr", 1e-6) / self.cfg.train.lr,
        )

        if self.is_bpgs:
            # SOTA Requirement: Decoupled Uncertainty Optimizer
            lr_unc = self.cfg.method.get("lr_theta", self.cfg.train.lr)
            opt_unc = torch.optim.AdamW(unc_params, lr=lr_unc, weight_decay=0.0)
            
            # Note: We return both optimizers. PyTorch Lightning manual optimization
            # allows us to step them independently in training_step.
            return [opt_net, opt_unc], [{"scheduler": sch_net, "interval": "step"}]
            
        return {
            "optimizer": opt_net,
            "lr_scheduler": {
                "scheduler": sch_net,
                "interval": "step",
                "frequency": 1,
            },
        }
