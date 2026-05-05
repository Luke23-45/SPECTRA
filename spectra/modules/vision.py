"""
spectra/modules/vision.py
-------------------------
Vertical Silo for the Vision Domain (NYUv2 Spatial Tasks).
Contains ZERO logic for Sepsis or MSE tabular tasks.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Dict, Any

from spectra.modules.base import OrthogonalSPECTRAModule
from spectra.engine.optimizers import OptimizationEngine
from spectra.architectures.builder import build_model
from spectra.engine.losses import LOSS_REGISTRY
from spectra.engine.weighters import build_weighter
from spectra.evaluation.metrics import (
    SegmentationMetrics,
    DepthMetrics,
    NormalMetrics
)
from spectra.utils.optimizer import build_optimizer_and_scheduler
from spectra.data.nyuv2.transforms import NYUv2BatchTrainTransform
import logging

logger = logging.getLogger("spectra.vision")

class VisionSPECTRAModule(OrthogonalSPECTRAModule):
    def __init__(self, cfg: DictConfig, engine: OptimizationEngine):
        super().__init__(cfg, engine)
        
        # 1. Architecture Assembly
        self.model = build_model(cfg)
        
        self.backbone = self.model.backbone
        self.heads = self.model.heads
        # 2. Weighter Initialization
        self.weighter = build_weighter(cfg)
        method_name = cfg.get("method_name") or cfg.get("method", {}).get("name")
        self.is_pcgrad = (method_name == "pcgrad")
        self.is_bpgs = (method_name == "bpgs")
        batch_aug_mode = cfg.get("batch_augmentation", cfg.get("dataset", {}).get("batch_augmentation", "disabled"))
        if (cfg.get("dataset_name") or cfg.get("dataset", {}).get("name")) == "nyuv2" and batch_aug_mode != "disabled":
            self.batch_train_transform = NYUv2BatchTrainTransform(
                normalize_rgb=cfg.get("normalize_rgb", cfg.get("dataset", {}).get("normalize_rgb", False))
            )
            self.batch_augmentation_mode = batch_aug_mode
        else:
            self.batch_train_transform = None
            self.batch_augmentation_mode = "disabled"
        
        # 3. Tasks & Spatial Metrics
        self.task_names = [task.name for task in cfg.tasks]
        self.task_weights = nn.ParameterDict()
        self.task_losses = nn.ModuleDict()
        
        self._val_metrics: Dict[str, Any] = {}
        self._val_losses = nn.ModuleDict()

        for task in cfg.tasks:
            name = task.name
            
            # Loss Setup
            self.task_weights[name] = nn.Parameter(torch.tensor(task.get("weight", 1.0)), requires_grad=False)
            
            exclude = ["name", "loss", "weight", "metrics", "type", "manifold", "target"]
            loss_kwargs = {k: v for k, v in task.items() if k not in exclude}
            
            loss_fn = LOSS_REGISTRY[task.loss](**loss_kwargs)
            self.task_losses[name] = loss_fn
            self._val_losses[name] = LOSS_REGISTRY[task.loss](**loss_kwargs)

            # Vision Metrics Initialization
            if name == "segmentation":
                n_classes = task.get("num_classes", 13)
                ignore_idx = task.get("ignore_index", 255)
                self._val_metrics[name] = SegmentationMetrics(n_classes, ignore_index=ignore_idx)
            elif name == "depth":
                self._val_metrics[name] = DepthMetrics()
            elif name == "normals":
                self._val_metrics[name] = NormalMetrics()

    def forward(self, batch: Dict) -> Dict:
        return self.model(batch["input"])

    def on_after_batch_transfer(self, batch: Dict, dataloader_idx: int) -> Dict:
        if self.batch_train_transform is None:
            return batch
        if not getattr(self.trainer, "training", False):
            return batch

        inputs = batch.get("input")
        if not isinstance(inputs, torch.Tensor):
            return batch

        if self.batch_augmentation_mode == "cuda" and not inputs.is_cuda:
            return batch

        return self.batch_train_transform(batch)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        
        loss_dict = {}
        weighted_task_loss_list = []
        
        # Robustly extract targets from batch
        targets = batch.get("targets", batch.get("target"))
        
        for name in self.task_names:
            pred = predictions[name]
            target = targets[name]
            
            # Shape alignment
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)

            loss_fn = self.task_losses[name]
            if hasattr(loss_fn, 'requires_mask') and loss_fn.requires_mask and "meta" in batch:
                loss = loss_fn(pred, target, batch["meta"])
            else:
                loss = loss_fn(pred, target)
                
            weighted_task_loss_list.append(loss * self.task_weights[name])
            loss_dict[name] = loss

        losses_tensor = torch.stack(weighted_task_loss_list)
        raw_losses_tensor = torch.stack([loss_dict[n] for n in self.task_names])
        
        if self.is_pcgrad or self.is_bpgs:
            total_loss = losses_tensor.sum()
        else:
            total_loss, w_metrics = self.weighter(
                losses_tensor,
                shared_params=list(self.backbone.parameters()),
                sync_ddp=self.trainer.world_size > 1 if getattr(self, "trainer", None) else False,
                raw_losses=raw_losses_tensor,
            )
            bsz = batch.get("input").shape[0] if isinstance(batch.get("input"), torch.Tensor) else 1
            for key, val in w_metrics.items():
                self.log(f"train/{key}", val, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        
        final_loss = self.engine.backward_and_step(
            module=self,
            batch_idx=batch_idx,
            losses=loss_dict,
            total_loss=total_loss,
            optimizers=self.optimizers(),
            lr_schedulers=self.lr_schedulers()
        )

        bsz = batch.get("input").shape[0] if isinstance(batch.get("input"), torch.Tensor) else 1
        if final_loss is not None:
             self.log("train/total_loss", final_loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bsz)
        for name, loss in loss_dict.items():
             self.log(f"train/{name}_loss", loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)

        return final_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch)
        total_val_loss = torch.tensor(0.0, device=self.device)
        
        # Robustly extract targets from batch
        targets = batch.get("targets", batch.get("target"))
        
        for name in self.task_names:
            pred = predictions[name]
            target = targets[name]

            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)

            val_loss_fn = self._val_losses[name]
            if hasattr(val_loss_fn, 'requires_mask') and val_loss_fn.requires_mask and "meta" in batch:
                loss = val_loss_fn(pred, target, batch["meta"])
            else:
                loss = val_loss_fn(pred, target)
                
            self.log(f"val/{name}_loss", loss, sync_dist=True, prog_bar=False, batch_size=batch.get("input").shape[0] if isinstance(batch.get("input"), torch.Tensor) else 1)
            total_val_loss = total_val_loss + loss * self.task_weights[name]

            if name in self._val_metrics:
                metric_obj = self._val_metrics[name]
                if isinstance(metric_obj, SegmentationMetrics):
                    metric_obj.update(pred, target)
                elif isinstance(metric_obj, DepthMetrics):
                    mask = batch.get("meta", {}).get("depth_mask", None)
                    metric_obj.update(pred, target, mask=mask)
                elif isinstance(metric_obj, NormalMetrics):
                    metric_obj.update(pred, target)

        self.log("val/total_loss", total_val_loss, sync_dist=True, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        for m in self._val_metrics.values():
            m.reset()

    def on_validation_epoch_end(self) -> None:
        if not self._val_metrics: return

        for name, metric_obj in self._val_metrics.items():
            if isinstance(metric_obj, SegmentationMetrics):
                r = metric_obj.compute()
                self.log(f"val/{name}_miou", r["miou"], prog_bar=True, sync_dist=False)
                self.log(f"val/{name}_pixel_acc", r["pixel_acc"], prog_bar=False, sync_dist=False)
                if name == "segmentation":
                    self.log("val/miou", r["miou"], prog_bar=True, sync_dist=False)

            elif isinstance(metric_obj, DepthMetrics):
                r = metric_obj.compute()
                self.log(f"val/{name}_abs_rel", r["abs_rel"], prog_bar=True, sync_dist=False)
                self.log(f"val/{name}_rmse", r["rmse"], prog_bar=False, sync_dist=False)
                if name == "depth":
                    self.log("val/depth_abs_rel", r["abs_rel"], prog_bar=True, sync_dist=False)

            elif isinstance(metric_obj, NormalMetrics):
                r = metric_obj.compute()
                self.log(f"val/{name}_mean_angle", r["mean_angle_deg"], prog_bar=True, sync_dist=False)
                self.log(f"val/{name}_within_11_25", r["within_11_25"], prog_bar=False, sync_dist=False)

        if hasattr(self.weighter, "get_telemetry"):
            tel = self.weighter.get_telemetry()
            for i, lv in enumerate(tel.get("log_vars", [])):
                self.log(f"val/bpgs_log_var_{i}", lv, sync_dist=False)

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.cfg)
