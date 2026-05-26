"""
Orthogonal engine API protocol.
Ensures optimization logic is completely decoupled from domain/dataset logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch
import pytorch_lightning as pl

class OptimizationEngine(ABC):
    """
    Abstract Base Class for Orthogonal Optimization Engines.
    
    The Engine takes raw loss tensors from the Domain Reactor and 
    executes the backward pass, optimizer stepping, and metric logging.
    It has zero knowledge of where the losses came from (Clinical vs Vision).
    """

    @abstractmethod
    def setup(self, module: pl.LightningModule) -> None:
        """
        Called once during module initialization.
        Sets flags like `module.automatic_optimization`.
        """
        pass

    @abstractmethod
    def backward_and_step(
        self,
        module: pl.LightningModule,
        batch_idx: int,
        losses: Dict[str, torch.Tensor],
        total_loss: torch.Tensor,
        optimizers: Any,
        lr_schedulers: Any
    ) -> torch.Tensor:
        """
        Executes the backward pass and steps the optimizer/scheduler.
        
        Args:
            module: The parent LightningModule (used for `module.log()` telemetry).
            batch_idx: The current batch index.
            losses: A dictionary mapping task names to scalar loss tensors.
            total_loss: The sum of `losses`, used by standard engines.
            optimizers: The Lightning optimizer object(s).
            lr_schedulers: The Lightning LR scheduler object(s).
            
        Returns:
            The final scalar loss tensor (for logging purposes).
        """
        pass
