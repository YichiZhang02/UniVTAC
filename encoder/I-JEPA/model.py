"""Single-frame image JEPA with visible context and masked spatial targets."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from vit_encoder import TactileViT
from vit_common import decoder
from .sigreg import SIGReg


class IJEPAEncoder(TactileViT):
    def forward(self, images):
        return self.patch_tokens(images).mean(dim=1)


class IJEPA(nn.Module):
    def __init__(self, size, in_chans=3, block_size=6, num_blocks=2,
                 variance_weight=0.1, min_feature_std=0.1,
                 uniformity_weight=0.0, variance_scope='full',
                 sigreg_weight=0.0, sigreg_dim=128,
                 sigreg_projections=256):
        super().__init__()
        self.encoder = IJEPAEncoder(size, in_chans=in_chans)
        self.teacher = copy.deepcopy(self.encoder)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.variance_weight = variance_weight
        self.min_feature_std = min_feature_std
        self.uniformity_weight = uniformity_weight
        self.sigreg_weight = sigreg_weight
        if variance_scope not in ('full', 'context'):
            raise ValueError('variance_scope must be full or context')
        self.variance_scope = variance_scope
        if num_blocks not in (1, 2):
            raise ValueError('num_blocks must be 1 or 2')
        if num_blocks * block_size ** 2 >= self.encoder.num_patches:
            raise ValueError('Target blocks leave no visible context patches')
        positions = torch.cartesian_prod(torch.arange(self.encoder.grid_size - block_size + 1),
                                         torch.arange(self.encoder.grid_size - block_size + 1))
        self.register_buffer('block_positions', positions, persistent=False)
        if num_blocks == 2:
            valid = ((positions[:, None, 0] - positions[None, :, 0]).abs() >= block_size
                     ) | ((positions[:, None, 1] - positions[None, :, 1]).abs() >= block_size)
            pairs = torch.nonzero(valid)
            if not len(pairs):
                raise ValueError('No room for two non-overlapping target blocks')
            self.register_buffer('block_pairs', pairs, persistent=False)
        dim = self.encoder.embed_dim
        if sigreg_weight:
            self.sigreg_projector = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, sigreg_dim))
            self.sigreg = SIGReg(projections=sigreg_projections)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 196, dim))
        self.predictor = decoder(dim, depth=2, heads=6 if size == 'S' else 12)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, images):
        batch, count = images.shape[0], self.encoder.num_patches
        grid = self.encoder.grid_size
        if self.num_blocks == 1:
            starts = [self.block_positions[torch.randint(len(self.block_positions),
                                                          (batch,), device=images.device)]]
        else:
            pair = self.block_pairs[torch.randint(len(self.block_pairs), (batch,), device=images.device)]
            starts = [self.block_positions[pair[:, index]] for index in range(2)]
        row = torch.arange(grid, device=images.device)[None, :, None]
        col = torch.arange(grid, device=images.device)[None, None, :]
        mask = torch.zeros(batch, grid, grid, dtype=torch.bool, device=images.device)
        for block_start in starts:
            mask |= ((row >= block_start[:, 0, None, None])
                     & (row < block_start[:, 0, None, None] + self.block_size)
                     & (col >= block_start[:, 1, None, None])
                     & (col < block_start[:, 1, None, None] + self.block_size))
        mask = mask.flatten(1)
        visible = (~mask).int().argsort(dim=1, descending=True)[
            :, :count - self.num_blocks * self.block_size ** 2]
        context = self.encoder.tokens(images, visible)[:, 1:]
        full = self.mask_token.to(context.dtype).expand(batch, count, -1).clone()
        full.scatter_(1, visible.unsqueeze(-1).expand_as(context), context)
        prediction = self.predictor(full + self.pos)
        with torch.no_grad():
            target = self.teacher.patch_tokens(images)
            target = F.layer_norm(target, (target.shape[-1],))
        prediction = F.layer_norm(prediction, (prediction.shape[-1],))
        prediction_loss = F.mse_loss(prediction[mask], target[mask])
        # The EMA target can become constant across images while the masked
        # prediction loss continues to fall. Keep both pooled and spatial
        # student features informative across the batch.
        student_tokens = (self.encoder.patch_tokens(images) if self.variance_scope == 'full'
                          else context).float()
        spatial_std = student_tokens.std(dim=0, unbiased=False)
        pooled_std = student_tokens.mean(dim=1).std(dim=0, unbiased=False)
        if self.variance_scope == 'full':
            variance_loss = (F.relu(self.min_feature_std - spatial_std).mean()
                             + F.relu(self.min_feature_std - pooled_std).mean()) / 2
        else:
            variance_loss = F.relu(self.min_feature_std - pooled_std).mean()
        if batch > 1:
            pooled = F.normalize(student_tokens.mean(dim=1), dim=-1)
            distances = torch.cdist(pooled, pooled).masked_fill(
                torch.eye(batch, device=images.device, dtype=torch.bool), float('inf'))
            uniformity_loss = -distances.min(dim=1).values.clamp_min(1e-6).log().mean()
        else:
            uniformity_loss = prediction_loss.new_zeros(())
        if self.sigreg_weight and batch > 1:
            # The sample axis is the batch of images. A learned projection
            # carries the Gaussian constraint without forcing every raw ViT
            # channel to be standard normal.
            gaussian_features = self.sigreg_projector(context.float().mean(dim=1))
            sigreg_loss = self.sigreg(gaussian_features.unsqueeze(0))
            projected_std = gaussian_features.std(dim=0, unbiased=False).mean()
        else:
            sigreg_loss = prediction_loss.new_zeros(())
            projected_std = prediction_loss.new_zeros(())
        loss = (prediction_loss + self.variance_weight * variance_loss
                + self.uniformity_weight * uniformity_loss
                + self.sigreg_weight * sigreg_loss)
        target_std = target.float().std(dim=0).mean()
        return loss, {'masked_latent': prediction_loss.detach(),
                      'variance_regularizer': variance_loss.detach(),
                      'uniformity_regularizer': uniformity_loss.detach(),
                      'sigreg': sigreg_loss.detach(),
                      'sigreg_projected_std': projected_std.detach(),
                      'student_spatial_std': spatial_std.mean().detach(),
                      'student_pooled_std': pooled_std.mean().detach(),
                      'masked_fraction': mask.float().mean().detach(),
                      'target_feature_std': target_std.detach()}

    @torch.no_grad()
    def update_teacher(self, momentum=.996):
        for student, teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            teacher.lerp_(student, 1 - momentum)
