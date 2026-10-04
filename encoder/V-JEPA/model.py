"""Action-conditioned, EMA-target tactile JEPA with LeWM-style SIGReg."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from importlib import import_module

from vit_encoder import TactileViT

SIGReg = import_module('I-JEPA.sigreg').SIGReg


class VJEPAEncoder(TactileViT):
    """Projected CLS latent; patch tokens remain available for ACT full mode."""
    def __init__(self, size, in_chans=3):
        super().__init__(size, in_chans=in_chans)
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
        )

    def forward(self, images):
        return self.projector(self.tokens(images)[:, 0])


class VJEPA(nn.Module):
    def __init__(self, size, in_chans=3, action_dim=8, sigreg_weight=0.09,
                 sigreg_projections=256, predictor_depth=2):
        super().__init__()
        self.encoder = VJEPAEncoder(size, in_chans=in_chans)
        self.teacher = copy.deepcopy(self.encoder)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        dim = self.encoder.embed_dim
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim + action_dim),
            nn.Linear(dim + action_dim, dim * 2), nn.GELU(),
            *sum(([nn.Linear(dim * 2, dim * 2), nn.GELU()] for _ in range(max(0, predictor_depth - 1))), []),
            nn.Linear(dim * 2, dim),
        )
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg(projections=sigreg_projections)

    def forward(self, current, future, action):
        context = self.encoder(current)
        prediction = self.predict_future_latent(context, action)
        with torch.no_grad():
            target = self.teacher(future)
        prediction = prediction.float()
        target = target.float()
        prediction_loss = F.mse_loss(prediction, target)
        latent = context.float()
        sigreg_loss = self.sigreg(latent.unsqueeze(0)) if len(latent) > 1 else prediction_loss.new_zeros(())
        loss = prediction_loss + self.sigreg_weight * sigreg_loss
        return loss, {
            'future_latent': prediction_loss.detach(),
            'sigreg': sigreg_loss.detach(),
            'sigreg_latent_std': latent.std(dim=0, unbiased=False).mean().detach(),
            'context_feature_std': context.float().std(dim=0, unbiased=False).mean().detach(),
            'target_feature_std': target.std(dim=0, unbiased=False).mean().detach(),
        }

    def predict_future_latent(self, context_latent, action):
        """Predict a future latent from an encoded context and normalized action."""
        return self.predictor(torch.cat((context_latent, action.to(context_latent.dtype)), dim=-1))

    @torch.no_grad()
    def update_teacher(self, momentum=.996):
        for student, teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            teacher.lerp_(student, 1 - momentum)
