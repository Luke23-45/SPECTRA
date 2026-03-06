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
        # [SOTA FIX] OnlineTargetScaler Normalization Matching
        # BPGS MUST track the normalized losses so that its variance
        # measurement matches the exact scale of the network loss applied.
        ema_losses = [losses[name] for name in module.task_names]
        module.weighter.update_ema(ema_losses)

        # ── 2. Base Flow — Network weights ───────────────────────────────────
        # CRITICAL: network_loss() MUST use the NORMALIZED losses (from the
        # original computation graph) so gradients flow correctly through the
        # model. Only update_ema() above uses the raw un-normalized losses.
        raw_opt_net.zero_grad()
        net_losses = [losses[name] for name in module.task_names]
        loss_net = module.weighter.network_loss(net_losses)

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
            # SOTA TITANIUM FIX 2: Universal Decoupled Parameter Gradient Clipping
            # BPGS explicitly forces expert/high-freq task gradients to scale massively.
            # Using global norm on the entire model squashes unscaled generative trunks to exactly 0.
            # We must clip EVERY architectural component completely independently.
            components_to_clip = []
            if hasattr(module, 'backbone') and module.backbone is not None:
                components_to_clip.append(module.backbone)
            if hasattr(module, 'alb') and module.alb is not None:
                components_to_clip.append(module.alb)
            if hasattr(module, 'heads') and module.heads is not None:
                # Clip each head's parameters independently as well
                for head in module.heads.values():
                    components_to_clip.append(head)

            if len(components_to_clip) > 0:
                for comp in components_to_clip:
                    torch.nn.utils.clip_grad_norm_(comp.parameters(), max_norm=grad_clip)
            else:
                # Fallback if architecture doesn't follow expected topology
                torch.nn.utils.clip_grad_norm_(
                    [p for g in raw_opt_net.param_groups for p in g['params']],
                    max_norm=grad_clip,
                )

        net_step_overflow = False
        if scaler is not None:
            # Check if gradients overflowed (inf/nan) after unscaling.
            # If so, scaler.step() skips the step internally, but we MUST know this
            # so we can synchronize the Fiber flow constraint.
            opt_net_state = scaler._per_optimizer_states.get(id(raw_opt_net))
            if opt_net_state is not None:
                # 3 == inf/nan found in this optimizer
                if opt_net_state.get('found_inf_per_device', {}):
                    for found in opt_net_state['found_inf_per_device'].values():
                        if found.item() > 0:
                            net_step_overflow = True
                            break

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
            # SOTA TITANIUM FIX 3: Dual-Optimizer AMP Desynchronization Shielding
            # If the network overflowed to nan/inf, scaler.step(raw_opt_net) was suppressed.
            # If we allow the Fiber flow (theta params, Pure Fp32) to step now, BPGS 
            # shifts its probabilistic belief off an imagined network topology that never occurred.
            # They permanently desynchronize. We MUST freeze the uncertainty step.
            if not net_step_overflow:
                scaler.step(raw_opt_unc)
            else:
                # We skip stepping the optimizer but MUST manually clear the Fiber gradients.
                # Otherwise they accumulate infinitely.
                raw_opt_unc.zero_grad(set_to_none=True)

            # Called ONCE, after ALL optimizers have been stepped this iteration.
            # PyTorch docs: "scaler.update should only be called once, after all
            # optimizers used this iteration have been stepped."
            scaler.update()
        else:
            if not net_step_overflow:
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

        # [INVERSION FIX] Log the BPGS precision-weighted loss as a SEPARATE
        # diagnostic metric. This value rises as exp(-s_i) grows — expected
        # and healthy, but NOT comparable to val/total_loss.
        module.log('train/bpgs_weighted_loss', loss_net.detach(),
                   on_step=False, on_epoch=True, sync_dist=True)

        # [INVERSION FIX] Return UNWEIGHTED sum of task losses for logging
        # parity with val/total_loss (which uses losses_tensor.sum()).
        # The precision-weighted loss (loss_net) was already consumed by
        # .backward() above — it is NOT the quantity to compare against
        # validation loss, because its scale changes as BPGS adapts.
        raw_loss_sum = sum(losses[name] for name in module.task_names).detach()
        return raw_loss_sum
