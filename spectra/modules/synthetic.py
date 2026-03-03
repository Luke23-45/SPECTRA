"""
spectra/modules/synthetic.py
----------------------------
Vertical Silo for Synthetic/Small-Scale Domains.
Pure MSE loss regressions — extremely simple module.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Dict

from spectra.modules.base import OrthogonalSPECTRAModule
from spectra.engine.optimizers import OptimizationEngine
from spectra.architectures.builder import build_model
from spectra.engine.losses import LOSS_REGISTRY
from spectra.engine.weighters import build_weighter
from spectra.utils.optimizer import build_optimizer_and_scheduler

class SyntheticSPECTRAModule(OrthogonalSPECTRAModule):
    def __init__(self, cfg: DictConfig, engine: OptimizationEngine):
        super().__init__(cfg, engine)
        
        self.model = build_model(cfg)
        self.backbone = self.model.backbone
        self.heads = self.model.heads
        self.alb = getattr(self.model, 'alb', None)
        self.use_alb = self.alb is not None

        self.weighter = build_weighter(cfg)
        self.is_pcgrad = (cfg.get("method_name") == "pcgrad" or 
                          cfg.get("method", {}).get("name") == "pcgrad")
        
        self.task_names = [task.name for task in cfg.tasks]
        self.task_weights = nn.ParameterDict()
        self.task_losses = nn.ModuleDict()

        for task in cfg.tasks:
            name = task.name
            self.task_weights[name] = nn.Parameter(torch.tensor(task.get("weight", 1.0)), requires_grad=False)
            
            # Filter out non-loss arguments (Hydra/SPECTRA specific)
            exclude = ["name", "loss", "weight", "metrics", "type", "manifold", "target"]
            loss_kwargs = {k: v for k, v in task.items() if k not in exclude}
            self.task_losses[name] = LOSS_REGISTRY[task.loss](**loss_kwargs)

    def forward(self, batch: Dict) -> Dict:
        return self.model(batch["input"])

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = {}
        weighted_task_loss_list = []
        
        for name in self.task_names:
            pred = predictions[name]
            target = batch["targets"][name]
            
            # [SOTA Fix] Shape alignment for 2D batches (B, 1) vs (B,)
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)

            loss = self.task_losses[name](pred, target)
            weighted_task_loss_list.append(loss * self.task_weights[name])
            loss_dict[name] = loss

        losses_tensor = torch.stack(weighted_task_loss_list)
        
        if self.is_pcgrad:
            total_loss = losses_tensor.sum()
        else:
            shared_params = list(self.backbone.parameters())
            if self.use_alb:
                shared_params += list(self.alb.parameters())
            total_loss, w_metrics = self.weighter(
                losses_tensor,
                shared_params=shared_params,
                sync_ddp=self.trainer.world_size > 1 if getattr(self, "trainer", None) else False,
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
        
        return final_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch)
        total_val_loss = torch.tensor(0.0, device=self.device)
        
        for name in self.task_names:
            pred = predictions[name]
            target = batch["targets"][name]

            # [SOTA Fix] Shape alignment
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)

            loss = self.task_losses[name](pred, target)
            self.log(f"val/{name}_loss", loss, sync_dist=True)
            total_val_loss = total_val_loss + loss * self.task_weights[name]

        self.log("val/total_loss", total_val_loss, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.cfg)
