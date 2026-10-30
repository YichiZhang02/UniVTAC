"""Scratch SD-style convolutional VAE with 8x spatial compression.

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
from reconstruction_loss import reconstruction_mse


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
    embed_dim = 4
    grid_size = 28

    def __init__(self, size, in_chans=3, *, autoencoder=None):
        super().__init__()
        model = AutoencoderKL(**sd_config(size, in_chans)) if autoencoder is None else autoencoder
        self.encoder = model.encoder
        self.quant_conv = model.quant_conv

    def posterior(self, images):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(images * 2 - 1)))

    def latent_tokens(self, images):
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, (224, 224), mode='bilinear', align_corners=False)
        return self.posterior(images).mode().flatten(2).transpose(1, 2)

    patch_tokens = latent_tokens

    def cls_and_latent_tokens(self, images):
        patches = self.latent_tokens(images)
        return patches.mean(1, keepdim=True), patches

    def forward(self, images):
        return self.latent_tokens(images).mean(1)


class SDVAE(nn.Module):
    def __init__(self, size, in_chans=3, kl_weight=1e-4, foreground_weight=50.):
        super().__init__()
        model = AutoencoderKL(**sd_config(size, in_chans))
        self.encoder = SDVAEEncoder(size, in_chans, autoencoder=model)
        self.decoder = model.decoder
        self.post_quant_conv = model.post_quant_conv
        self.kl_weight = kl_weight
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.

    def forward(self, images):
        posterior = self.encoder.posterior(images)
        latent = posterior.sample() if self.training else posterior.mode()
        prediction = (self.decoder(self.post_quant_conv(latent)) + 1) / 2
        reconstruction = reconstruction_mse(prediction, images, self.foreground_weight)
        kl = posterior.kl().mean() / posterior.mean[0].numel()
        return reconstruction + self.kl_weight * kl, dict(reconstruction=reconstruction.detach(), kl=kl.detach())
