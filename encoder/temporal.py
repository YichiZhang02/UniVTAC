"""Four-frame spatial/temporal encoders and clip reconstruction helpers.

ViTs compress raw clips with tubelet Conv3d before joint space-time attention.
CNNs compress feature grids with temporal Conv3d after per-frame extraction.
Clip reconstruction helpers are adapted from TacMind encoder/common/temporal.py.
The time axis is oldest -> newest; output has one or two compressed time slices.
Single-frame scratch architectures do not depend on this module.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

from encoder.vit_encoder import TactileViT
from encoder.vit_common import patchify as image_patchify

NUM_FRAMES = 4
FRAME_STRIDE = 1
SIMULATION_STEP_STRIDE = 2


def require_clip(images):
    if images.ndim != 5 or images.shape[1] != NUM_FRAMES:
        raise ValueError(f'Expected [B,4,C,H,W], got {tuple(images.shape)}')


class TemporalViT(TactileViT):
    """Conv3d tubelets followed by joint attention over compressed time and space."""
    num_frames = NUM_FRAMES
    frame_stride = FRAME_STRIDE
    simulation_step_stride = SIMULATION_STEP_STRIDE
    temporal_backend = 'conv3d'

    def __init__(self, size='S', image_size=224, in_chans=3, temporal_compression='4in2'):
        super().__init__(size, image_size=image_size, in_chans=in_chans)
        self.output_frames = compression_frames(temporal_compression)
        self.temporal_compression = temporal_compression
        self.tubelet_size = NUM_FRAMES // self.output_frames
        self.vit.patch_embed = nn.Conv3d(in_chans, self.embed_dim,
            kernel_size=(self.tubelet_size, 16, 16), stride=(self.tubelet_size, 16, 16), bias=False)
        self.time_position = nn.Parameter(torch.empty(1, self.output_frames, 1, self.embed_dim))
        nn.init.trunc_normal_(self.time_position, std=.02)

    def tokens_slices(self, images, keep_indices=None, patch_mask=None, mask_token=None,
               *, single_frame_target=False):
        # Repeat one future observation only across the tubelet kernel. This
        # shares Conv3d weights with history and produces exactly one target slice.
        if single_frame_target:
            if images.ndim != 4:
                raise ValueError('A future single-frame target must be [B,C,H,W]')
            images = images[:, None].expand(-1, self.tubelet_size, -1, -1, -1)
        else:
            require_clip(images)
        b, t, c, h, w = images.shape
        images = images.reshape(b * t, c, h, w)
        if (h, w) != (self.image_size, self.image_size):
            images = F.interpolate(images, (self.image_size, self.image_size), mode='bilinear', align_corners=False)
        images = images.reshape(b, t, c, self.image_size, self.image_size).transpose(1, 2)
        x = self.vit.patch_embed(images).flatten(3).permute(0, 2, 3, 1)
        t = x.shape[1]
        if patch_mask is not None:
            x = torch.where(patch_mask[:, None, :, None], mask_token.to(x.dtype), x)
        pos = self.vit.pos_embed[:, 1:]
        if keep_indices is not None:
            index = keep_indices[:, None, :, None].expand(-1, t, -1, self.embed_dim)
            x = x.gather(2, index)
            pos = pos.expand(b, -1, -1).gather(1, keep_indices[..., None].expand(-1, -1, self.embed_dim))
        x = x + pos[:, None]
        cls = (self.vit.cls_token + self.vit.pos_embed[:, :1]).expand(b, -1, -1)
        cls = cls[:, None].expand(-1, t, -1, -1)
        x = torch.cat((cls, x), dim=2) + self.time_position[:, -t:]
        x = self.vit.pos_drop(x)
        n = x.shape[2]
        # Keep a CLS for each slice for DINO/V-JEPA objectives. All slice CLS
        # and patch tokens attend jointly; tokens() exports one averaged CLS.
        x = x.reshape(b, t * n, self.embed_dim)
        for block in self.vit.blocks:
            x = block(x)
        return self.vit.norm(x).reshape(b, t, n, self.embed_dim)

    def tokens(self, images, keep_indices=None, patch_mask=None, mask_token=None):
        x = self.tokens_slices(images, keep_indices, patch_mask, mask_token)
        return torch.cat((x[:, :, :1].mean(1), x[:, :, 1:].flatten(1, 2)), 1)


class TemporalGrid(nn.Module):
    """Compress four CNN feature grids using non-overlapping temporal kernels."""
    def __init__(self, dim, temporal_compression='4in2'):
        super().__init__()
        self.output_frames = compression_frames(temporal_compression)
        tubelet_size = NUM_FRAMES // self.output_frames
        self.compression = nn.Conv3d(dim, dim, kernel_size=(tubelet_size, 1, 1),
                                     stride=(tubelet_size, 1, 1))

    def forward(self, grids):
        b, t, d, h, w = grids.shape
        if t != NUM_FRAMES:
            raise ValueError('TemporalGrid requires four frames')
        return self.compression(grids.transpose(1, 2)).transpose(1, 2)


def compression_frames(mode):
    if mode not in ('4in1', '4in2'):
        raise ValueError(f'Unsupported temporal compression: {mode}')
    return int(mode[-1])


class TimeExpansion(nn.Module):
    """Expand K compressed spatial grids to four frames, with learned time positions.

    4in2 uses the first slice for frames 0/1 and the second for frames 2/3.
    No uncompressed encoder features bypass the bottleneck.
    """
    def __init__(self, dim, temporal_compression='4in2'):
        super().__init__()
        self.output_frames = compression_frames(temporal_compression)
        self.position = nn.Parameter(torch.empty(1, NUM_FRAMES, 1, dim))
        nn.init.trunc_normal_(self.position, std=.02)

    def forward(self, tokens):
        b, count, d = tokens.shape
        if count % self.output_frames:
            raise ValueError('Compressed tokens must contain K equally sized spatial grids')
        tokens = tokens.reshape(b, self.output_frames, -1, d)
        tokens = tokens.repeat_interleave(NUM_FRAMES // self.output_frames, dim=1)
        return (tokens + self.position).flatten(0, 1)


def patchify(images, patch_size=16):
    require_clip(images)
    # Fold time into batch for shared image losses and decoders.
    return image_patchify(images.flatten(0, 1), patch_size)


def augment(images, local=False):
    """Independent views, with crop/brightness shared by all frames in a clip."""
    require_clip(images)
    output = []
    for clip in images:
        scale = (.55, .8) if local else (.8, 1.)
        i, j, h, w = RandomResizedCrop.get_params(clip[-1], scale, (.85, 1.15))
        crop = TF.resized_crop(clip, i, j, h, w, [224, 224])
        crop = TF.adjust_brightness(crop, float(torch.empty(()).uniform_(.9, 1.1)))
        output.append(crop.clamp(0, 1))
    return torch.stack(output)
