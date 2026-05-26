"""Vertical silo for the clinical domain (Sepsis/EHR)."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Dict, Any
import logging
from torchmetrics import AUROC, AveragePrecision, Recall, Accuracy

logger = logging.getLogger(__name__)

from spectra.modules.base import OrthogonalSPECTRAModule
from spectra.engine.optimizers import OptimizationEngine
from spectra.architectures.builder import build_model
from spectra.engine.losses import LOSS_REGISTRY
from spectra.engine.weighters import build_weighter
from spectra.utils.optimizer import build_optimizer_and_scheduler

class ClinicalSPECTRAModule(OrthogonalSPECTRAModule):
    def __init__(self, cfg: DictConfig, engine: OptimizationEngine):
        super().__init__(cfg, engine)
        

        self.model = build_model(cfg)
        

        self.backbone = self.model.backbone
        self.heads = self.model.heads

        self.weighter = build_weighter(cfg)
        method_name = cfg.get("method_name") or cfg.get("method", {}).get("name")
        self.is_pcgrad = (method_name == "pcgrad")
        self.is_bpgs = (method_name == "bpgs")
        

        self.task_names = [task.name for task in cfg.tasks]
        self.task_weights = nn.ParameterDict()
        self.task_losses = nn.ModuleDict()
        

        self.train_metrics = nn.ModuleDict()
        self._val_metrics = nn.ModuleDict()

        for task in cfg.tasks:
            name = task.name
            

            self.task_weights[name] = nn.Parameter(torch.tensor(task.get("weight", 1.0)), requires_grad=False)
            exclude = ["name", "loss", "weight", "metrics", "type", "manifold", "target", "output_dim"]
            loss_kwargs = {k: v for k, v in task.items() if k not in exclude}
            
            if 'pos_weight' in loss_kwargs and loss_kwargs['pos_weight'] is not None:
                loss_kwargs['pos_weight'] = torch.tensor([loss_kwargs['pos_weight']])
            
            loss_fn = LOSS_REGISTRY[task.loss](**loss_kwargs)
            self.task_losses[name] = loss_fn


            if name == "outcome":
                self.train_metrics[f"{name}_auc"] = AUROC(task="binary")
                self._val_metrics[f"{name}_auc"] = AUROC(task="binary")
                self._val_metrics[f"{name}_prc"] = AveragePrecision(task="binary")
                self._val_metrics[f"{name}_recall"] = Recall(task="binary")
            elif name == "phase":
                self.train_metrics[f"{name}_acc"] = Accuracy(task="multiclass", num_classes=task.get("num_classes", 3))
                self._val_metrics[f"{name}_acc"] = Accuracy(task="multiclass", num_classes=task.get("num_classes", 3))

    def forward(self, batch: Dict) -> Dict:
        return self.model(batch["input"])

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:

        predictions = self(batch)
        

        loss_dict = {}
        weighted_task_loss_list = []
        

        targets = batch.get("targets", batch.get("target"))
        
        for name in self.task_names:
            pred = predictions[name]
            target_clean = targets[name]
            target = target_clean.clone() # Keep clean copy for metrics
            
            # Shape alignment for 2D batches (B, 1) vs (B,)
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)
                
            loss = self.task_losses[name](pred, target)
            weighted_task_loss_list.append(loss * self.task_weights[name])
            loss_dict[name] = loss
            

            if name == "outcome":
                p_metric = pred.squeeze(-1) if pred.dim() > 1 and pred.shape[-1] == 1 else pred
                t_metric = target_clean.squeeze(-1) if target_clean.dim() > 1 and target_clean.shape[-1] == 1 else target_clean
                self.train_metrics[f"{name}_auc"].update(torch.sigmoid(p_metric), t_metric.long())
            elif name == "phase":
                p_metric = pred.squeeze(-1) if pred.dim() > 1 and pred.shape[-1] == 1 else pred
                t_metric = target_clean.squeeze(-1) if target_clean.dim() > 1 and target_clean.shape[-1] == 1 else target_clean
                self.train_metrics[f"{name}_acc"].update(p_metric, t_metric)

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

             self.log("step_loss", final_loss.detach(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=False, batch_size=bsz)
             self.log("train/total_loss", final_loss.detach(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        
        for name, loss in loss_dict.items():
             self.log(f"train/{name}_loss", loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)

        return final_loss

    def on_train_epoch_end(self) -> None:
        for name, metric in self.train_metrics.items():
            # Use try/except for safe metric compute
            try:
                val = metric.compute()
                self.log(f"train/{name.split('_')[1].upper()}", val, prog_bar=True, sync_dist=True)
            except (RuntimeError, ValueError) as getattr_err:
                logger.debug(f"[Epoch {self.current_epoch}] Metric {name} skipped: No positive samples or no updates.") 
            finally:
                metric.reset()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch)
        weighted_task_loss_list = []
        

        targets = batch.get("targets", batch.get("target"))
        
        for name in self.task_names:
            pred = predictions[name]
            target = targets[name]
            
            # Shape alignment for 2D batches (B, 1) -> (B,)
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)
                
            loss = self.task_losses[name](pred, target)
            self.log(f"val/{name}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            weighted_task_loss_list.append(loss * self.task_weights[name])
            
            if name == "outcome":
                self._val_metrics[f"{name}_auc"].update(torch.sigmoid(pred), target.long())
                self._val_metrics[f"{name}_prc"].update(torch.sigmoid(pred), target.long())
                self._val_metrics[f"{name}_recall"].update(torch.sigmoid(pred), target.long())
            elif name == "phase":
                self._val_metrics[f"{name}_acc"].update(pred, target)

        losses_tensor = torch.stack(weighted_task_loss_list)

        # val/total_loss uses plain weighted sum (not uncertainty weighter)
        # to keep the early stopping signal scale-consistent across epochs.
        total_val_loss = losses_tensor.sum()

        self.log("val/total_loss", total_val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        # Compute and log all metrics, then reset to free memory.
        for name, metric in self._val_metrics.items():
            try:
                val = metric.compute()
                # e.g., 'outcome_auc' -> 'outcome_AUC'
                parts = name.split('_')
                if len(parts) >= 2:
                    log_name = f"val/{parts[0]}_{parts[1].upper()}"
                else:
                    log_name = f"val/{name.upper()}"
                
                self.log(log_name, val, prog_bar=True, sync_dist=True)
            except (RuntimeError, ValueError) as err:
                logger.debug(f"[Validation Epoch {self.current_epoch}] Metric {name} skipped: {str(err)}")
            finally:
                metric.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        return build_optimizer_and_scheduler(self, self.cfg)
