"""PCGrad manual gradient surgery engine."""

from typing import Dict, Any, List
import torch
import torch.distributed as dist
import pytorch_lightning as pl

from spectra.engine.optimizers.base import OptimizationEngine

class PCGradEngine(OptimizationEngine):
    """
    Executes PCGrad gradient surgery via manual optimization.
    Requires `module.automatic_optimization = False`.
    """

    def setup(self, module: pl.LightningModule) -> None:
        module.automatic_optimization = False

    def backward_and_step(
        self,
        module: pl.LightningModule,
        batch_idx: int,
        losses: Dict[str, torch.Tensor],
        total_loss: torch.Tensor,
        optimizers: Any,
        lr_schedulers: Any
    ) -> torch.Tensor:
        opt = optimizers if not isinstance(optimizers, list) else optimizers[0]
        sch = lr_schedulers if not isinstance(lr_schedulers, list) else lr_schedulers[0] if lr_schedulers else None
        
        opt.zero_grad()

        scaler = getattr(module.trainer.precision_plugin, "scaler", None)
        raw_opt = opt.optimizer if hasattr(opt, "optimizer") else opt
        
        # Resolve shared parameters from the common backbone.
        shared_params = list(module.backbone.parameters())
            
        # The losses passed in are already weighted by `module.task_weights`
        weighted_task_loss_list = list(losses.values())

        # 1. Per-task backbone gradients (SCALED magnitude if AMP)
        task_grads = []
        for i, task_loss in enumerate(weighted_task_loss_list):
            loss_to_backprop = scaler.scale(task_loss) if scaler is not None else task_loss
            grads = torch.autograd.grad(
                loss_to_backprop, shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            task_grads.append(grads)

        # 2. PCGrad surgery + assignment
        pcgrad_metrics = module.weighter.project_and_assign(task_grads, shared_params)

        # 3. Head gradients: each head sees ONLY its own task loss
        for idx, (task_name, task_loss) in enumerate(losses.items()):
            if task_name in module.heads:
                head_params = list(module.heads[task_name].parameters())
                if not head_params:
                    continue
                
                is_last_head = (idx == len(losses) - 1)
                loss_to_backprop = scaler.scale(task_loss) if scaler is not None else task_loss
                head_grads = torch.autograd.grad(
                    loss_to_backprop, head_params,
                    retain_graph=not is_last_head,  # Free graph on the last head
                    allow_unused=True,
                )
                for p, g in zip(head_params, head_grads):
                    if g is not None:
                        p.grad = g

        # 4. Unscale ALL gradients (backbone + heads) uniformly if AMP
        if scaler is not None:
            scaler.unscale_(raw_opt)

        # DDP explicit synchronization:
        # autograd.grad bypasses DDP backward hooks, so PCGrad
        # gradients are purely local. Manual all-reduce is required.
        if module.trainer.world_size > 1 and torch.distributed.is_initialized():
            for p in module.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        # 5. Gradient clipping
        if module.cfg.train.get("grad_clip", 0) > 0:
            module.clip_gradients(opt, gradient_clip_val=module.cfg.train.grad_clip)

        # 6. Optimizer Step
        if scaler is not None:
            # NaN-safe optimizer step
            old_scale = scaler.get_scale()
            scaler.step(raw_opt)
            scaler.update()

            # Scheduler step (skip if scaler reduced scale to avoid bad LR update)
            if sch is not None and scaler.get_scale() >= old_scale:
                sch.step()
        else:
            opt.step()
            if sch is not None:
                sch.step()

        # pcgrad/total_conflicts is still logged here as it's engine-specific
        module.log("pcgrad/total_conflicts", pcgrad_metrics.get("pcgrad/total_conflicts", 0), prog_bar=True, on_step=True)
        
        if sch is not None:
             module.log("lr", sch.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)

        return None # Return None for manual optimization
