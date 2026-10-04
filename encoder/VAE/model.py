"""Spatial VAE with 49 latent tokens and same-input reconstruction."""
import torch
from torch import nn
from torch.nn import functional as F
from vit_encoder import TactileViT
from vit_common import decoder, patchify
from reconstruction_loss import reconstruction_mse


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


class VAEEncoder(nn.Module):
    def __init__(self, size, in_chans=3):
        super().__init__()
        self.vit = TactileViT(size, in_chans=in_chans)
        self.embed_dim = self.vit.embed_dim
        self.downsample = ResidualDownsample(self.embed_dim)
        self.mu = nn.Conv2d(self.embed_dim, self.embed_dim, 1)
        self.logvar = nn.Conv2d(self.embed_dim, self.embed_dim, 1)

    def parameters_for_latent(self, images):
        patches = self.vit.patch_tokens(images)
        b, _, d = patches.shape
        grid = patches.transpose(1, 2).reshape(b, d, 14, 14)
        latent = self.downsample(grid)
        mu = self.mu(latent).flatten(2).transpose(1, 2)
        logvar = self.logvar(latent).flatten(2).transpose(1, 2).clamp(-10, 10)
        return mu, logvar

    def latent_tokens(self, images):
        return self.parameters_for_latent(images)[0]

    def forward(self, images):
        return self.latent_tokens(images).mean(dim=1)


class VAE(nn.Module):
    def __init__(self, size, in_chans=3, kl_weight=1e-4, foreground_weight=50.):
        super().__init__()
        self.encoder = VAEEncoder(size, in_chans)
        self.kl_weight = kl_weight
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.
        d = self.encoder.embed_dim
        self.upsample = ResidualUpsample(d)
        self.to_decoder = nn.Linear(d, 256)
        self.pos = nn.Parameter(torch.zeros(1, 196, 256))
        self.decode = decoder(256)
        self.pixels = nn.Linear(256, 16 * 16 * in_chans)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, images):
        mu, logvar = self.encoder.parameters_for_latent(images)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        b, _, d = z.shape
        grid = z.transpose(1, 2).reshape(b, d, 7, 7)
        upsampled = self.upsample(grid)
        tokens = upsampled.flatten(2).transpose(1, 2)
        patches = self.pixels(self.decode(self.to_decoder(tokens) + self.pos))
        recon = reconstruction_mse(patches, patchify(images), self.foreground_weight)
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return recon + self.kl_weight * kl, {'reconstruction': recon.detach(), 'kl': kl.detach()}
