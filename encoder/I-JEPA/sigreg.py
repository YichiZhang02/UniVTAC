"""Sketched isotropic Gaussian regularization from LeJEPA / LeWM.

For each view, compare the empirical characteristic function of random
one-dimensional projections with that of N(0, 1). The sample axis is the
batch, so spatial position alone cannot hide image-level collapse.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SIGReg(nn.Module):
    def __init__(self, knots=17, projections=256):
        super().__init__()
        if knots < 2 or projections < 1:
            raise ValueError('SIGReg needs at least two knots and one projection')
        self.projections = projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        trapezoid = torch.full((knots,), 2 * dt, dtype=torch.float32)
        trapezoid[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        self.register_buffer('t', t)
        self.register_buffer('gaussian_cf', gaussian_cf)
        self.register_buffer('weights', trapezoid * gaussian_cf)

    def forward(self, embeddings):
        """embeddings: [views, batch, dimension]."""
        if embeddings.ndim != 3 or embeddings.shape[1] < 2:
            raise ValueError('SIGReg expects [views, batch>=2, dimension]')
        z = embeddings.float()
        directions = F.normalize(torch.randn(z.shape[-1], self.projections,
                                               device=z.device, dtype=z.dtype), dim=0)
        projected = z @ directions  # [views, batch, projections]
        phases = projected.unsqueeze(-1) * self.t
        cosine = phases.cos().mean(dim=1)
        sine = phases.sin().mean(dim=1)
        error = (cosine - self.gaussian_cf).square() + sine.square()
        return ((error * self.weights).sum(dim=-1) * z.shape[1]).mean()
