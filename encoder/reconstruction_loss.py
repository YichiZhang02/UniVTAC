"""Pixel reconstruction losses shared by tactile encoder pretraining methods."""
import torch
from torch.nn import functional as F


def reconstruction_mse(prediction, target, foreground_weight=0., mask=None):
    """Emphasize sparse marker dots while preserving the selected input target."""
    if not foreground_weight and mask is None:
        return F.mse_loss(prediction, target)
    weights = 1 + foreground_weight * target if foreground_weight else torch.ones_like(target)
    if mask is not None:
        weights = weights * mask
    return ((prediction - target).square() * weights).sum() / weights.sum()
