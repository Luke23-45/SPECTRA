"""
spectra/engine/losses.py
------------------------
Decoupled Loss Function Registry for SPECTRA.
"""

import torch
import torch.nn as nn

def _build_bce_loss(pos_weight=None, **kwargs):
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]))
    return nn.BCEWithLogitsLoss()

def _build_ce_loss(ignore_index=-100, **kwargs):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)

def _build_masked_l1_loss(**kwargs):
    from spectra.engine.dense_losses import MaskedL1Loss
    return MaskedL1Loss()

def _build_dense_cosine_loss(**kwargs):
    from spectra.engine.dense_losses import DenseCosineLoss
    return DenseCosineLoss()

LOSS_REGISTRY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "bce": _build_bce_loss,
    "binary_cross_entropy": _build_bce_loss,
    "cross_entropy": _build_ce_loss,
    "cosine": nn.CosineEmbeddingLoss,
    "masked_l1": _build_masked_l1_loss,
    "cosine_dense": _build_dense_cosine_loss,
}
