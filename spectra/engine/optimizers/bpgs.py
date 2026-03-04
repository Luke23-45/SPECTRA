"""
spectra/engine/optimizers/bpgs.py
---------------------------------
B-PGS Engine (Manual Decoupled Optimization).
Implements the SOTA bifurcated Flow architecture (Base vs Fiber) 
for the Synthetic/Orthogonal pipeline.
"""

from typing import Dict, Any
import torch
import torch.distributed as dist
import pytorch_lightning as pl

from spectra.engine.optimizers.base import OptimizationEngine

class BPGSEngine(OptimizationEngine):
    """
    Executes decoupled dual-optimizer B-PGS learning.
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
        opts = optimizers if isinstance(optimizers, list) else [optimizers]
        opt_net = opts[0]
        opt_unc = opts[1] if len(opts) > 1 else None
        
        sch_net = lr_schedulers[0] if isinstance(lr_schedulers, list) and len(lr_schedulers) > 0 else lr_schedulers
        sch_unc = lr_schedulers[1] if isinstance(lr_schedulers, list) and len(lr_schedulers) > 1 else None
        scaler = getattr(module.trainer.precision_plugin, "scaler", None)
        
        # 1. Update EMA tracker
        weighted_task_loss_list = list(losses.values())
        module.weighter.update_ema(weighted_task_loss_list)
        
        # 2. Base Flow (Network weights)
        opt_net.zero_grad()
        loss_net = module.weighter.network_loss(weighted_task_loss_list)
        
        if scaler is not None:
            module.manual_backward(scaler.scale(loss_net))
            scaler.unscale_(opt_net)
        else:
            module.manual_backward(loss_net)
            
        if getattr(module.cfg.train, "grad_clip", 0) > 0:
            module.clip_gradients(opt_net, gradient_clip_val=module.cfg.train.grad_clip)
            
        if scaler is not None:
            scaler.step(opt_net)
        else:
            opt_net.step()
            
        # 3. Fiber Flow (Uncertainty parameters)
        opt_unc.zero_grad()
        loss_unc = module.weighter.uncertainty_loss()
        
        if scaler is not None:
            module.manual_backward(scaler.scale(loss_unc))
            scaler.unscale_(opt_unc)
        else:
            module.manual_backward(loss_unc)
            
        # DDP explicit synchronization for Weighter parameters
        if module.trainer.world_size > 1 and dist.is_initialized():
            for p in module.weighter.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        if scaler is not None:
            scaler.step(opt_unc)
            scaler.update()
        else:
            opt_unc.step()
            
        # 4. Schedule step
        if sch_net is not None:
            sch_net.step()
            module.log("lr", sch_net.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)
            
        if sch_unc is not None:
            sch_unc.step()
            module.log("lr_unc", sch_unc.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)

        # Output Telemetry
        w_metrics = module.weighter.get_task_stats()
        bsz = 1 # We can't easily grab bsz here, PL will handle mean on log later
        for key, val in w_metrics.items():
            module.log(f"train/{key}", val, on_step=False, on_epoch=True, sync_dist=True)

        return None # Manual optimization returns None
