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
from spectra.engine.scalers import OnlineTargetScaler
from spectra.evaluation.metrics import (
    RegressionMetrics,
    BinaryClassificationMetrics,
    MultiLabelClassificationMetrics,
)

class SyntheticSPECTRAModule(OrthogonalSPECTRAModule):
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
        self.task_types = {task.name: task.get("type", "regression") for task in cfg.tasks}
        self.task_weights = nn.ParameterDict()
        self.task_losses = nn.ModuleDict()
        self.target_scalers = nn.ModuleDict()
        self._val_regression_metrics = {}
        self._val_binary_metrics = {}
        self._classification_task_names = []
        self._val_multilabel_metrics = None

        for task in cfg.tasks:
            name = task.name
            self.task_weights[name] = nn.Parameter(torch.tensor(task.get("weight", 1.0)), requires_grad=False)
            
            if task.get("type", "regression") == "regression":
                self.target_scalers[name] = OnlineTargetScaler()
                self._val_regression_metrics[name] = RegressionMetrics()
            elif task.get("type") == "classification" and task.get("loss") in {"bce", "binary_cross_entropy"}:
                self._val_binary_metrics[name] = BinaryClassificationMetrics()
                self._classification_task_names.append(name)
            
            # Filter out non-loss arguments (Hydra/SPECTRA specific)
            exclude = ["name", "loss", "weight", "metrics", "type", "manifold", "target", "output_dim"]
            loss_kwargs = {k: v for k, v in task.items() if k not in exclude}
            self.task_losses[name] = LOSS_REGISTRY[task.loss](**loss_kwargs)

        dataset_name = cfg.get("dataset_name") or cfg.get("dataset", {}).get("name")
        if dataset_name == "yeast" and self._classification_task_names:
            self._val_multilabel_metrics = MultiLabelClassificationMetrics(
                num_labels=len(self._classification_task_names)
            )

    def forward(self, batch: Dict) -> Dict:
        return self.model(batch["input"])

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = {}
        raw_loss_dict = {}
        weighted_task_loss_list = []
        
        for name in self.task_names:
            pred = predictions[name]
            target = batch["targets"][name]
            
            # [SOTA Fix] Shape alignment for 2D batches (B, 1) vs (B,)
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)

            # [SOTA Fix] Online Target Normalization to prevent Adam scale trap
            if name in self.target_scalers:
                target_for_loss = self.target_scalers[name].normalize(target)
                
                # [SOTA Fix] Create raw unscaled metrics for training to match Validation logic
                # using .detach() to ensure no gradients flow backward from the logging block.
                pred_unscaled = self.target_scalers[name].denormalize(pred.detach())
                raw_l = self.task_losses[name](pred_unscaled, target)
            else:
                target_for_loss = target
                raw_l = self.task_losses[name](pred.detach(), target)

            loss = self.task_losses[name](pred, target_for_loss)
            weighted_task_loss_list.append(loss * self.task_weights[name])
            loss_dict[name] = loss
            raw_loss_dict[name] = raw_l

        losses_tensor = torch.stack(weighted_task_loss_list)
        raw_losses_list = [loss_dict[n] for n in self.task_names]
        raw_losses_tensor = torch.stack(raw_losses_list)
        
        if self.is_pcgrad:
            total_loss = losses_tensor.sum()
        elif self.is_bpgs:
            # BPGS: network_loss for base flow (gradients to model)
            total_loss = self.weighter.network_loss(raw_losses_list)
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

        # BPGS: also compute uncertainty_loss for fiber flow (gradients to theta)
        if self.is_bpgs:
            uncertainty = self.weighter.uncertainty_loss(raw_losses_list)
            # Note: uncertainty_loss backward is handled by the optimizer on theta
            # which is separate from the model parameter optimization
            total_loss = total_loss + uncertainty

        final_loss = self.engine.backward_and_step(
            module=self,
            batch_idx=batch_idx,
            losses=loss_dict,
            total_loss=total_loss,
            optimizers=self.optimizers(),
            lr_schedulers=self.lr_schedulers()
        )

        bsz = batch.get("input").shape[0] if isinstance(batch.get("input"), torch.Tensor) else 1
        
        # [SOTA Fix] Ensure training loss is ALWAYS logged to progress bar, 
        # even during manual optimization where final_loss is None.
        log_loss = final_loss.detach() if final_loss is not None else total_loss.detach()
        self.log("train/total_loss", log_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bsz)
        
        # Log BOTH normalized mathematical loss and raw human-readable loss
        for name, loss in loss_dict.items():
            self.log(f"train/{name}_loss_norm", loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        for name, raw in raw_loss_dict.items():
            self.log(f"train/{name}_loss", raw.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)        
        return final_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch)
        weighted_task_loss_list = []
        multilabel_logits = []
        multilabel_targets = []
        
        for name in self.task_names:
            pred = predictions[name]
            target = batch["targets"][name]

            # [SOTA Fix] Shape alignment
            if pred.dim() <= 2 and target.dim() <= 2:
                if pred.dim() > target.dim(): pred = pred.squeeze(-1)
                if target.dim() > pred.dim(): target = target.squeeze(-1)

            # [SOTA Fix] Dual-Scale Metric resolution
            # 1. Denormalize for human-readable metrics (RAW real-world error)
            # 2. Normalize target for optimizer scaling (NORMALIZED error)
            if name in self.target_scalers:
                pred_unscaled = self.target_scalers[name].denormalize(pred)
                target_norm = self.target_scalers[name].normalize(target)
            else:
                pred_unscaled = pred
                target_norm = target

            # Raw human-readable loss logging (Scale: ~80,000)
            raw_loss = self.task_losses[name](pred_unscaled, target)
            self.log(f"val/{name}_loss", raw_loss, sync_dist=True)

            metric_obj = self._val_regression_metrics.get(name)
            if metric_obj is not None:
                metric_obj.update(pred_unscaled, target)
            binary_metric_obj = self._val_binary_metrics.get(name)
            if binary_metric_obj is not None:
                binary_metric_obj.update(pred, target)
                multilabel_logits.append(pred.reshape(-1, 1))
                multilabel_targets.append(target.reshape(-1, 1))
            
            # Normalized loss for dynamic scaling (Scale: ~1.0)
            norm_loss = self.task_losses[name](pred, target_norm)
            self.log(f"val/{name}_loss_norm", norm_loss, sync_dist=True)
            weighted_task_loss_list.append(norm_loss * self.task_weights[name])

        if self._val_multilabel_metrics is not None and multilabel_logits:
            logits = torch.cat(multilabel_logits, dim=1)
            targets = torch.cat(multilabel_targets, dim=1)
            self._val_multilabel_metrics.update(logits, targets)

        losses_tensor = torch.stack(weighted_task_loss_list)

        # CRITICAL: val/total_loss must NOT pass through the uncertainty weighter.
        # KendallWeighter.forward() injects training-evolved σ as precision weights.
        # As σ shrinks during training, 0.5/σ² grows even when task losses improve —
        # causing val/total_loss to rise while the model gets better, which corrupts
        # the early stopping signal.
        # Plain weighted sum is scale-consistent across all epochs and all methods.
        total_val_loss = losses_tensor.sum()

        self.log("val/total_loss", total_val_loss, sync_dist=True, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        for metric in self._val_regression_metrics.values():
            metric.reset()
        for metric in self._val_binary_metrics.values():
            metric.reset()
        if self._val_multilabel_metrics is not None:
            self._val_multilabel_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        if self._val_regression_metrics:
            rmse_values = []
            mae_values = []
            r2_values = []

            for name, metric_obj in self._val_regression_metrics.items():
                result = metric_obj.compute()
                self.log(f"val/{name}_rmse", result["rmse"], sync_dist=False)
                self.log(f"val/{name}_mae", result["mae"], sync_dist=False)
                self.log(f"val/{name}_r2", result["r2"], sync_dist=False)
                rmse_values.append(result["rmse"])
                mae_values.append(result["mae"])
                r2_values.append(result["r2"])

            mean_rmse = sum(rmse_values) / len(rmse_values)
            mean_mae = sum(mae_values) / len(mae_values)
            mean_r2 = sum(r2_values) / len(r2_values)

            self.log("val/rmse", mean_rmse, prog_bar=True, sync_dist=False)
            self.log("val/mae", mean_mae, prog_bar=False, sync_dist=False)
            self.log("val/r2", mean_r2, prog_bar=True, sync_dist=False)

        for name, metric_obj in self._val_binary_metrics.items():
            result = metric_obj.compute()
            self.log(f"val/{name}_f1", result["f1"], sync_dist=False)
            self.log(f"val/{name}_accuracy", result["accuracy"], sync_dist=False)
            self.log(f"val/{name}_precision", result["precision"], sync_dist=False)
            self.log(f"val/{name}_recall", result["recall"], sync_dist=False)

        if self._val_multilabel_metrics is not None:
            result = self._val_multilabel_metrics.compute()
            self.log("val/micro_f1", result["micro_f1"], prog_bar=True, sync_dist=False)
            self.log("val/macro_f1", result["macro_f1"], sync_dist=False)
            self.log("val/hamming_acc", result["hamming_acc"], prog_bar=False, sync_dist=False)
            self.log("val/subset_acc", result["subset_acc"], prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.cfg)
