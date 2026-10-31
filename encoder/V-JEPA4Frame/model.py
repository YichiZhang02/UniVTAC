"""Four-frame history to future-latent JEPA, with optional action conditioning."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from importlib import import_module

from encoder.temporal import TemporalViT as TactileViT

SIGReg = import_module('encoder.I-JEPA.sigreg').SIGReg


class VJEPAEncoder(TactileViT):
    """Four-frame ViT with one/two compressed time slices and a shared projector."""
    def __init__(self, size, in_chans=3, temporal_compression='4in2'):
        super().__init__(size, in_chans=in_chans, temporal_compression=temporal_compression)
        self.num_frames = 4
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
        )

    def forward(self, images):
        return self.projected_latent(images).reshape(images.shape[0], self.output_frames, self.embed_dim).mean(1)

    def tokens(self, images, **kwargs):
        x = self.project_tokens(super().tokens_slices(images, **kwargs))
        return torch.cat((x[:, :, :1].mean(1), x[:, :, 1:].flatten(1, 2)), 1)

    def project_tokens(self, tokens):
        # BatchNorm1d expects [samples, channels], not [B, tokens, channels].
        shape = tokens.shape
        return self.project_cls(tokens.reshape(-1, shape[-1])).reshape(shape)

    def projected_latent(self, images, *, single_frame_target=False):
        # Project the compressed CLS; future targets use the shared single-frame path.
        x = super().tokens_slices(images, single_frame_target=single_frame_target)[:, :, 0]
        return self.project_cls(x.flatten(0, 1)).reshape(x.shape[0], -1)

    def project_cls(self, x):
        for layer in self.projector:
            if isinstance(layer, nn.BatchNorm1d) and layer.training and x.shape[0] == 1:
                # File-chunk batches can end with one sample; use the accumulated stats.
                x = F.batch_norm(x, layer.running_mean, layer.running_var, layer.weight,
                                 layer.bias, training=False, eps=layer.eps)
            else:
                x = layer(x)
        return x


class VJEPA(nn.Module):
    def __init__(self, size, in_chans=3, action_dim=8, sigreg_weight=0.09,
                 sigreg_projections=256, predictor_depth=2, temporal_compression='4in2'):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = VJEPAEncoder(size, in_chans=in_chans, temporal_compression=temporal_compression)
        self.teacher = copy.deepcopy(self.encoder)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        dim = self.encoder.embed_dim
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim * self.encoder.output_frames + action_dim),
            nn.Linear(dim * self.encoder.output_frames + action_dim, dim * 2), nn.GELU(),
            *sum(([nn.Linear(dim * 2, dim * 2), nn.GELU()] for _ in range(max(0, predictor_depth - 1))), []),
            nn.Linear(dim * 2, dim),
        )
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg(projections=sigreg_projections)

    def forward(self, current, future, action=None):
        context = self.encoder.projected_latent(current)
        prediction = self.predict_future_latent(context, action)
        with torch.no_grad():
            target = self.teacher.projected_latent(future, single_frame_target=True)
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

    def predict_future_latent(self, context_latent, action=None):
        """Predict a future latent from an encoded context and normalized action."""
        if self.action_dim == 0:
            if action is not None and action.shape[-1] != 0:
                raise ValueError('This tactile-only predictor was trained without action inputs')
            return self.predictor(context_latent)
        if action is None or action.shape[-1] != self.action_dim:
            raise ValueError(f'Predictor requires {self.action_dim}-dimensional actions')
        return self.predictor(torch.cat((context_latent, action.to(context_latent.dtype)), dim=-1))

    def train(self, mode=True):
        super().train(mode)
        for name in ('teacher', 'teacher_cls_head', 'teacher_patch_head'):
            if hasattr(self, name):
                getattr(self, name).eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum=.996):
        for student, teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            teacher.lerp_(student, 1 - momentum)
        for student, teacher in zip(self.encoder.buffers(), self.teacher.buffers()):
            teacher.copy_(student)


Encoder = VJEPAEncoder
Model = VJEPA
