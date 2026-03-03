"""
spectra/modules/base.py
-----------------------
Base Class for Vertical Silos (Domain Modules).
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from spectra.engine.optimizers import OptimizationEngine

class OrthogonalSPECTRAModule(pl.LightningModule):
    """
    Hollow Orchestrator for SPECTRA.
    Delegates backpropagation to orthogonal initialized Engines.
    Overrides `optimizer_step` and `backward` to route through the Engine.
    """

    def __init__(self, cfg: DictConfig, engine: OptimizationEngine):
        super().__init__()
        self.cfg = cfg
        self.engine = engine
        # Let the engine set flags like `self.automatic_optimization = False`
        if self.engine is not None:
            self.engine.setup(self)

    # All domain-specific logic (models, losses, metrics, forward pass)
    # MUST be implemented in the subclasses (Clinical, Vision, Synthetic).
    
    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
