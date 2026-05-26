"""Standard automatic backpropagation engine."""

from typing import Dict, Any
import torch
import pytorch_lightning as pl

from spectra.engine.optimizers.base import OptimizationEngine

class StandardEngine(OptimizationEngine):
    """
    Executes standard PyTorch Lightning automatic optimization.
    Relies on `module.automatic_optimization = True`.
    """

    def setup(self, module: pl.LightningModule) -> None:
        module.automatic_optimization = True

    def backward_and_step(
        self,
        module: pl.LightningModule,
        batch_idx: int,
        losses: Dict[str, torch.Tensor],
        total_loss: torch.Tensor,
        optimizers: Any,
        lr_schedulers: Any
    ) -> torch.Tensor:
        # In automatic_optimization mode, Lightning handles the backward pass
        # and optimizer stepping automatically based on the returned tensor
        # from `training_step`. The Engine here merely passes the tensor back.
        
        # Log learning rate natively (Optional, LRMonitor usually handles this)
        if lr_schedulers is not None:
             sch = lr_schedulers if not isinstance(lr_schedulers, list) else lr_schedulers[0]
             module.log("lr", sch.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False)

        return total_loss
