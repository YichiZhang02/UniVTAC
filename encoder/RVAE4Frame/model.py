"""Spatial VAE with 49 latent tokens and four-frame reconstruction from one/two compressed latent grids."""
import torch
from torch import nn
from torch.nn import functional as F
from encoder.temporal import TemporalViT as TactileViT
from encoder.vit_common import decoder
from encoder.temporal import patchify, TimeExpansion
from encoder.reconstruction_loss import reconstruction_mse


class ResidualDownsample(nn.Module):
    """Learn the 14x14 -> 7x7 spatial reduction without averaging tokens."""
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
        )
        self.shortcut = nn.Conv2d(channels, channels, 1, stride=2, bias=False)
        self.activation = nn.GELU()

    def forward(self, grid):
        return self.activation(self.main(grid) + self.shortcut(grid))


class ResidualUpsample(nn.Module):
    """Learn the 7x7 -> 14x14 spatial expansion for reconstruction."""
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
        )
        self.shortcut = nn.Conv2d(channels, channels, 1, bias=False)
        self.activation = nn.GELU()

    def forward(self, grid):
        skip = self.shortcut(F.interpolate(grid, scale_factor=2, mode='nearest'))
        return self.activation(self.main(grid) + skip)


class RVAEEncoder(nn.Module):
    def __init__(self, size, in_chans=3, temporal_compression='4in2'):
        super().__init__()
        self.vit = TactileViT(size, in_chans=in_chans, temporal_compression=temporal_compression)
        self.temporal_compression = temporal_compression
        self.output_frames = int(temporal_compression[-1])
        self.num_frames = 4
        self.frame_stride = 1
        self.simulation_step_stride = 2
        self.embed_dim = self.vit.embed_dim
        self.output_frames = self.vit.output_frames
        self.downsample = ResidualDownsample(self.embed_dim)
        self.mu = nn.Conv2d(self.embed_dim, self.embed_dim, 1)
        self.logvar = nn.Conv2d(self.embed_dim, self.embed_dim, 1)

    def parameters_for_latent(self, images, *, vit_tokens=None):
        patches = self.vit.patch_tokens(images) if vit_tokens is None else vit_tokens[:, 1:]
        b, _, d = patches.shape
        grid = patches.reshape(b * self.output_frames, 196, d).transpose(1, 2).reshape(b * self.output_frames, d, 14, 14)
        latent = self.downsample(grid)
        mu = self.mu(latent).flatten(2).transpose(1, 2)
        logvar = self.logvar(latent).flatten(2).transpose(1, 2).clamp(-10, 10)
        return mu.reshape(b, -1, d), logvar.reshape(b, -1, d)

    def cls_and_latent_tokens(self, images):
        tokens = self.vit.tokens(images)
        mu, _ = self.parameters_for_latent(images, vit_tokens=tokens)
        return tokens[:, :1], mu

    def latent_tokens(self, images):
        return self.parameters_for_latent(images)[0]

    patch_tokens = latent_tokens

    def forward(self, images):
        return self.vit(images)


class RVAE(nn.Module):
    def __init__(self, size, in_chans=3, kl_weight=1e-4, foreground_weight=50., temporal_compression='4in2'):
        super().__init__()
        self.encoder = RVAEEncoder(size, in_chans, temporal_compression=temporal_compression)
        self.kl_weight = kl_weight
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.
        d = self.encoder.embed_dim
        self.upsample = ResidualUpsample(d)
        self.time_expansion = TimeExpansion(d, temporal_compression)
        self.to_decoder = nn.Linear(d, 256)
        self.pos = nn.Parameter(torch.zeros(1, 196, 256))
        self.decode = decoder(256)
        self.pixels = nn.Linear(256, 16 * 16 * in_chans)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, images):
        mu, logvar = self.encoder.parameters_for_latent(images)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        b, _, d = z.shape
        k = self.encoder.output_frames
        grid = z.reshape(b * k, 49, d).transpose(1, 2).reshape(b * k, d, 7, 7)
        upsampled = self.upsample(grid)
        tokens = upsampled.flatten(2).transpose(1, 2).reshape(b, k * 196, d)
        patches = self.pixels(self.decode(self.to_decoder(self.time_expansion(tokens)) + self.pos))
        recon = reconstruction_mse(patches, patchify(images), self.foreground_weight)
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return recon + self.kl_weight * kl, {'reconstruction': recon.detach(), 'kl': kl.detach()}


VAE = RVAE
VAEEncoder = RVAEEncoder


Encoder = RVAEEncoder
Model = RVAE
