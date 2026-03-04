"""
spectra/engine/optimizers/bpgs.py
---------------------------------
B-PGS Engine (Manual Decoupled Optimization).
Implements the SOTA bifurcated Flow architecture (Base vs Fiber) 
for the Synthetic/Orthogonal pipeline.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
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

        # ─── Extract raw PyTorch optimizers to bypass PL's LightningOptimizer
        # hooks. PL's hook calls scaler.update() on every .step(), which would
        # reset _per_optimizer_states and cause the AssertionError on our
        # explicit scaler.update() at the end of this method.
        raw_opt_net = getattr(opt_net, '_optimizer', opt_net)
        raw_opt_unc = getattr(opt_unc, '_optimizer', opt_unc) if opt_unc is not None else None

        sch_net = lr_schedulers[0] if isinstance(lr_schedulers, list) and len(lr_schedulers) > 0 else lr_schedulers
        sch_unc = lr_schedulers[1] if isinstance(lr_schedulers, list) and len(lr_schedulers) > 1 else None

        # Resolve scaler: prefer trainer precision plugin, fall back gracefully.
        scaler = getattr(module.trainer.precision_plugin, 'scaler', None)

        grad_clip = getattr(module.cfg.train, 'grad_clip', 0.0)

        # ── 1. Update EMA tracker (no grad required) ─────────────────────────
        weighted_task_loss_list = list(losses.values())
        module.weighter.update_ema(weighted_task_loss_list)

        # ── 2. Base Flow — Network weights ───────────────────────────────────
        raw_opt_net.zero_grad()
        loss_net = module.weighter.network_loss(weighted_task_loss_list)

        if scaler is not None:
            # Canonical PyTorch AMP: scale → backward → unscale → clip → step.
            # Do NOT use module.manual_backward(scaler.scale(loss)): that routes
            # through pre_backward which calls scaler.scale() a second time,
            # producing scale²·g gradients that overflow to inf in fp16.
            scaler.scale(loss_net).backward()
            scaler.unscale_(raw_opt_net)
        else:
            loss_net.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in raw_opt_net.param_groups for p in g['params']],
                max_norm=grad_clip,
            )

        if scaler is not None:
            scaler.step(raw_opt_net)   # raw optimizer → PL hook NOT triggered
        else:
            raw_opt_net.step()

        # ── 3. Fiber Flow — Uncertainty / weighting parameters ───────────────
        raw_opt_unc.zero_grad()
        loss_unc = module.weighter.uncertainty_loss()

        if scaler is not None:
            scaler.scale(loss_unc).backward()
            scaler.unscale_(raw_opt_unc)
        else:
            loss_unc.backward()

        # DDP explicit sync for weighter params (they live outside DDP model).
        if module.trainer.world_size > 1 and dist.is_initialized():
            for p in module.weighter.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        if scaler is not None:
            scaler.step(raw_opt_unc)
            # Called ONCE, after ALL optimizers have been stepped this iteration.
            # PyTorch docs: "scaler.update should only be called once, after all
            # optimizers used this iteration have been stepped."
            scaler.update()
        else:
            raw_opt_unc.step()

        # ── 4. LR Schedule step ──────────────────────────────────────────────
        if sch_net is not None:
            sch_net.step()
            module.log('lr', sch_net.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)

        if sch_unc is not None:
            sch_unc.step()
            module.log('lr_unc', sch_unc.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)

        # ── 5. Telemetry ─────────────────────────────────────────────────────
        for key, val in module.weighter.get_task_stats().items():
            module.log(f'train/{key}', val, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss.detach()
