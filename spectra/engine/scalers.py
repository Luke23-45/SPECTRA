"""
spectra/engine/scalers.py
-------------------------
State-of-the-Art (SOTA) Dynamic Target Standardization.

Resolves the Adam Scale-Invariance Trap mathematically in Multi-Task Learning.
Tracks target distributions (mean, variance) online using running
average, ensuring gradients are perfectly scaled regardless of extreme
target offsets (e.g., millions vs decimals).
"""

import torch
import torch.nn as nn

class OnlineTargetScaler(nn.Module):
    """
    Standardizes neural network targets on-the-fly dynamically.
    
    Training:
        $y_{norm} = (y - \mu) / \sigma$
        Ensures Network trains iteratively bounded in strictly symmetric scales.

    Validation/Inference:
        $\hat{y} = \hat{y}_{norm} \cdot \sigma + \mu$
        Inverse scales output predictions to absolute task domain values, 
        ensuring comparative metrics remain identically scaled.

    Uses momentum (default 0.1) for running statistics.
    Buffer persistence ensures these stats safely serialize in checkpoints.
    """
    def __init__(self, momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        
        # Persistent state for checkpoints
        self.register_buffer("running_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(1, dtype=torch.float32))
        # Initialized flag prevents zero-bias on the very first batch
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update(self, targets: torch.Tensor):
        """Update running statistics dynamically given a batch of targets."""
        batch_mean = targets.mean()
        batch_var = targets.var(unbiased=False) if targets.numel() > 1 else torch.zeros_like(batch_mean)

        # [SOTA Fix] Removed synchronous DDP all_reduce.
        # Calling all_reduce sequentially 14 times per batch completely locks the GPU.
        # In DDP, batches are IID (Independent & Identically Distributed) across GPUs,
        # so local running-average updates are perfectly unbiased and mathematically sufficient.

        if not self.initialized.item():
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
            self.initialized.fill_(True)
        else:
            # [SOTA Fix] Use in-place copy_ instead of reassignment to preserve buffer persistence
            new_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            new_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            self.running_mean.copy_(new_mean)
            self.running_var.copy_(new_var)

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Standardizes targets to Z-scores.
        If in training mode, it securely updates the statistics before normalizing.
        """
        if self.training:
            self.update(targets)
        
        std = torch.sqrt(self.running_var + self.eps)
        return (targets - self.running_mean) / std

    def denormalize(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Un-standardizes network predictions back to the absolute task scale.
        """
        std = torch.sqrt(self.running_var + self.eps)
        return preds * std + self.running_mean

