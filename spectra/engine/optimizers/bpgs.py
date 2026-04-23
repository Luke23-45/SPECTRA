"""
spectra/engine/optimizers/bpgs.py
---------------------------------
Canonical B-PGS engine for the active research path.

Implements the split optimization contract:
  - optimizer 0 updates network parameters
  - optimizer 1 updates B-PGS uncertainty parameters
"""

from typing import Any, Dict

import pytorch_lightning as pl
import torch

from spectra.engine.optimizers.base import OptimizationEngine


class BPGSEngine(OptimizationEngine):
    """Execute canonical B-PGS split optimization."""

    def setup(self, module: pl.LightningModule) -> None:
        module.automatic_optimization = False

    def backward_and_step(
        self,
        module: pl.LightningModule,
        batch_idx: int,
        losses: Dict[str, torch.Tensor],
        total_loss: torch.Tensor,
        optimizers: Any,
        lr_schedulers: Any,
    ) -> torch.Tensor:
        opts = optimizers if isinstance(optimizers, list) else [optimizers]
        opt_net = opts[0]
        opt_unc = opts[1] if len(opts) > 1 else None

        schs = lr_schedulers if isinstance(lr_schedulers, list) else [lr_schedulers]
        sch_net = schs[0] if len(schs) > 0 else None
        sch_unc = schs[1] if len(schs) > 1 else None

        raw_losses = [losses[name] for name in module.task_names]
        grad_clip = getattr(module.cfg.train, "grad_clip", 0.0)

        # Step 1: network flow
        opt_net.zero_grad()
        loss_net = module.weighter.network_loss(raw_losses)
        loss_net.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in opt_net.param_groups for p in group["params"]],
                max_norm=grad_clip,
            )
        opt_net.step()

        # Step 2: uncertainty flow
        if opt_unc is not None:
            opt_unc.zero_grad()
            loss_unc = module.weighter.uncertainty_loss(raw_losses)
            loss_unc.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(module.weighter.parameters(), max_norm=grad_clip)
            opt_unc.step()

        # Step 3: schedulers
        if sch_net is not None:
            sch_net.step()
            if hasattr(sch_net, "get_last_lr"):
                module.log("lr", sch_net.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)
        if sch_unc is not None:
            sch_unc.step()
            if hasattr(sch_unc, "get_last_lr"):
                module.log("lr_unc", sch_unc.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)

        # Step 4: telemetry
        for key, val in module.weighter.get_task_stats().items():
            module.log(f"train/{key}", val, on_step=False, on_epoch=True, sync_dist=True)

        return sum(raw_losses).detach()
