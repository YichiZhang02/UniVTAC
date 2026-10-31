"""Four-frame SD-style VAE with selectable 4-to-1/2 time and 8x spatial compression.

B uses the sd-vae-ft-mse widths (128, 256, 512, 512); S halves
the widths. Both retain two residual layers per stage, bottleneck attention,
four latent channels, and learned quantization/post-quantization projections.
Reference: https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/main/config.json
"""
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from encoder.temporal import TemporalGrid, TimeExpansion, require_clip
from encoder.reconstruction_loss import reconstruction_mse
from encoder.latent_patchify import patchify_latents


def sd_config(size, in_chans):
    if size not in ('S', 'B'):
        raise ValueError(f'SD-VAE supports S or B, got {size}')
    width = 64 if size == 'S' else 128
    return dict(in_channels=in_chans, out_channels=in_chans,
                down_block_types=('DownEncoderBlock2D',) * 4,
                up_block_types=('UpDecoderBlock2D',) * 4,
                block_out_channels=(width, 2*width, 4*width, 4*width),
                layers_per_block=2, latent_channels=4, norm_num_groups=32,
                sample_size=224, scaling_factor=.18215)


class SDVAEEncoder(nn.Module):
    def __init__(self, size, in_chans=3, *, autoencoder=None, temporal_compression='4in2', sd_patch_size=2):
        super().__init__()
        if sd_patch_size != 2:
            raise ValueError('SD-VAE requires sd_patch_size=2')
        self.sd_patch_size = sd_patch_size
        self.embed_dim = 4 * sd_patch_size**2
        self.grid_size = 28 // sd_patch_size
        model = AutoencoderKL(**sd_config(size, in_chans)) if autoencoder is None else autoencoder
        self.encoder = model.encoder
        self.quant_conv = model.quant_conv
        self.temporal_compression = temporal_compression
        self.output_frames = int(temporal_compression[-1])
        self.num_frames = 4
        self.frame_stride = 1
        self.simulation_step_stride = 2
        self.temporal = TemporalGrid(model.quant_conv.in_channels, temporal_compression)

    def posterior(self, images):
        require_clip(images)
        b, t, c, h, w = images.shape
        features = self.encoder(images.reshape(b * t, c, h, w) * 2 - 1)
        features = self.temporal(features.reshape(b, t, *features.shape[1:]))
        b, k, d, h, w = features.shape
        # Distribution batch dimension folds K; latent_tokens restores chronological slices.
        return DiagonalGaussianDistribution(self.quant_conv(features.reshape(b * k, d, h, w)))

    def latent_tokens(self, images):
        if images.shape[-2:] != (224, 224):
            b, t, c, h, w = images.shape
            images = F.interpolate(images.reshape(b * t, c, h, w), (224, 224), mode='bilinear', align_corners=False).reshape(b, t, c, 224, 224)
        latent = self.posterior(images).mode()
        _, c, h, w = latent.shape
        latent = latent.reshape(images.shape[0], self.output_frames, c, h, w).transpose(1, 2)
        return patchify_latents(latent, self.sd_patch_size)

    patch_tokens = latent_tokens

    def cls_and_latent_tokens(self, images):
        patches = self.latent_tokens(images)
        return patches.mean(1, keepdim=True), patches

    def forward(self, images):
        return self.latent_tokens(images).mean(1)


class SDVAE(nn.Module):
    def __init__(self, size, in_chans=3, kl_weight=1e-4, foreground_weight=50., temporal_compression='4in2', sd_patch_size=2):
        super().__init__()
        model = AutoencoderKL(**sd_config(size, in_chans))
        self.encoder = SDVAEEncoder(size, in_chans, autoencoder=model,
                                   temporal_compression=temporal_compression, sd_patch_size=sd_patch_size)
        self.decoder = model.decoder
        self.post_quant_conv = model.post_quant_conv
        self.time_expansion = TimeExpansion(4, temporal_compression)
        self.kl_weight = kl_weight
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.

    def forward(self, images):
        posterior = self.encoder.posterior(images)
        latent = posterior.sample() if self.training else posterior.mode()
        bk, d, h, w = latent.shape
        b = images.shape[0]
        tokens = self.time_expansion(latent.flatten(2).transpose(1, 2).reshape(b, -1, d))
        expanded = tokens.transpose(1, 2).reshape(b * 4, d, h, w)
        prediction = (self.decoder(self.post_quant_conv(expanded)) + 1) / 2
        reconstruction = reconstruction_mse(prediction, images.flatten(0, 1), self.foreground_weight)
        kl = posterior.kl().mean() / posterior.mean[0].numel()
        return reconstruction + self.kl_weight * kl, dict(reconstruction=reconstruction.detach(), kl=kl.detach())


Encoder = SDVAEEncoder
Model = SDVAE
