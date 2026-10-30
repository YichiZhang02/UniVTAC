"""Full-clip Wan training, preserving the reference streaming inference path.

For a five-frame clip, causal convolutions are equivalent to the streaming
1+4 execution. A temporal downsampler preserves the first frame unchanged,
then applies its valid stride-two convolution to the full clip. The residual
shortcut's left temporal padding already implements the same boundary rule.
Parameter names, shapes, dtypes and the five-frame observation contract stay
unchanged. Compilation fuses normalization/activation/backward operations.
"""
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan, WanEncoder3d, WanResample, patchify,
)


class FullClipWanResample(WanResample):
    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            return super().forward(x, feat_cache, [0] if feat_idx is None else feat_idx)
        b, c, t, h, w = x.shape
        x = self.resample(x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w))
        x = x.view(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        if t > 1:
            if t % 2 != 1:
                raise ValueError('Full-clip Wan temporal downsampling requires an odd frame count')
            x = torch.cat((x[:, :, :1], self.time_conv(x)), dim=2)
        return x


def _encoder_forward(encoder, x):
    return WanEncoder3d.forward(encoder, x, feat_cache=None)


_compiled_encoder_forward = torch.compile(_encoder_forward, fullgraph=True, dynamic=False)


class CompiledWanEncoder3d(WanEncoder3d):
    def forward(self, x, feat_cache=None, feat_idx=None):
        if self.training and torch.is_grad_enabled() and feat_cache is None:
            return _compiled_encoder_forward(self, x)
        return super().forward(x, feat_cache, [0] if feat_idx is None else feat_idx)


class FullClipWanAutoencoder(AutoencoderKLWan):
    def _encode(self, x):
        if (not self.training or not torch.is_grad_enabled() or x.shape[2] != 5
                or self.use_tiling or self.use_slicing):
            return super()._encode(x)
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
        return self.quant_conv(self.encoder(x))


def enable_wan_training_fast(model, *, compile_encoder=True, channels_last=True):
    """Opt in before DDP construction; state_dict and optimizer keys are stable."""
    if type(model) not in (AutoencoderKLWan, FullClipWanAutoencoder):
        raise TypeError('Expected the Diffusers AutoencoderKLWan implementation')
    model.__class__ = FullClipWanAutoencoder
    for layer in model.encoder.modules():
        if isinstance(layer, WanResample) and layer.mode == 'downsample3d':
            layer.__class__ = FullClipWanResample
    if compile_encoder:
        model.encoder.__class__ = CompiledWanEncoder3d
    if channels_last:
        torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    return model
