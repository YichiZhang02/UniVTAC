"""Differentiable external encoders. Constructor metadata supports offline policy restore."""
import importlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from encoder.pretrained_registry import ROOT, SOURCES, source_dir, temporal_spec, validate_identity
from encoder.catalog import canonical_name, resolve_identity


def resize_frames(x, hw):
    shape = x.shape
    return F.interpolate(x.reshape(-1, *shape[-3:]), hw, mode='bilinear',
                         align_corners=False, antialias=True).reshape(*shape[:-2], *hw)


def patchify_latents(latent, patch_size=2):
    """Pack adjacent latent pixels into channels; keep both latent time positions."""
    b, c, t, h, w = latent.shape
    if h % patch_size or w % patch_size:
        raise ValueError('Latent height and width must be divisible by patch size')
    return (latent.reshape(b, c, t, h//patch_size, patch_size, w//patch_size, patch_size)
            .permute(0, 2, 3, 5, 1, 4, 6).reshape(b, t*(h//patch_size)*(w//patch_size),
                                                 c*patch_size*patch_size))


class ExternalEncoder(nn.Module):
    def __init__(self, mode, method, size, *, metadata=None, pretrained=True, with_decoder=False):
        super().__init__()
        validate_identity(mode, method, size)
        public_method, public_size = method, size
        mode, method, size = resolve_identity(mode, method, size)
        self.mode, self.method, self.size = mode, method, size
        self.num_frames, self.frame_stride = temporal_spec(mode, method)
        folder = source_dir(mode, method)
        # A checkpoint may also contain tensors and optimizer state: constructor
        # metadata must stay small, JSON-serializable and independent of those weights.
        meta = {key: value for key, value in (metadata or {}).items()
                if key in ('model_config', 'source_repo', 'source_revision', 'format_version', 'wan_patch_size',
                           'initialization', 'latent_calibration')}
        self.mean, self.std = [0., 0., 0.], [1., 1., 1.]
        name = canonical_name(mode, method)
        builder = importlib.import_module(f'encoder.{name}.model').build
        builder(self, folder, meta, metadata, pretrained, with_decoder, method, size)
        self.metadata = dict(meta, pretrained_mode=mode, method=public_method, size=public_size,
                             num_frames=self.num_frames, frame_stride=self.frame_stride,
                             image_size=list(self.hw), in_chans=3, color_order='RGB',
                             source_repo=SOURCES[mode, method][1])
        if mode == 'tactile':
            self.metadata['background_offset'] = 130/255 if method == 'AnyTouch2' else .5
        if pretrained and (folder / 'source.json').exists():
            self.metadata['source_revision'] = json.loads((folder / 'source.json').read_text())['revision']

    def preprocess(self, x):
        expected = 5 if self.num_frames > 1 else 4
        if x.ndim != expected or (expected == 5 and x.shape[1] != self.num_frames):
            raise ValueError(f'{self.mode}/{self.method} expects ' +
                             (f'[B,{self.num_frames},C,H,W]' if expected == 5 else '[B,C,H,W]') + f', got {tuple(x.shape)}')
        if self.mode == 'tactile' and self.method != 'AnyTouch2':
            # UniVTAC sensors are landscape; upstream Sparsh rotates them to portrait.
            x = torch.rot90(x, k=-1, dims=(-2, -1))
        x = resize_frames(x, self.hw)
        shape = [1] * x.ndim
        shape[-3] = 3
        return (x - x.new_tensor(self.mean).reshape(shape)) / x.new_tensor(self.std).reshape(shape)

    def posterior(self, x):
        x = self.preprocess(x).transpose(1, 2)
        # New inputs contain five real chronological observations. Only legacy
        # four-frame checkpoints need the original repeated-first-frame adapter.
        if self.num_frames == 4:
            x = torch.cat((x[:, :, :1], x), 2)
        return self.model.encode(x).latent_dist

    def patch_tokens(self, x):
        if self.method == 'Wan2.2-VAE':
            latent = self.posterior(x).mode()
            mean = latent.new_tensor(self.model.config.latents_mean)[None, :, None, None, None]
            std = latent.new_tensor(self.model.config.latents_std)[None, :, None, None, None]
            latent = (latent-mean)/std
            return (patchify_latents(latent, self.wan_patch_size) if self.wan_patch_size > 1
                    else latent.flatten(2).transpose(1, 2))
        x = self.preprocess(x)
        if self.mode == 'tactile' and self.method != 'AnyTouch2':
            # Sparsh's official dataset emits newest-to-oldest clips.
            x = x.flip(1).transpose(1, 2) if self.method == 'V-JEPA' else x.flip(1).flatten(1, 2)
            return self.model.forward_features(x)['x_norm_patchtokens']
        if self.method == 'AnyTouch2':
            return self.model(x)[:, 6:]
        if self.method == 'V-JEPA2':
            return self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        output = self.model(pixel_values=x).last_hidden_state
        if self.method == 'DINOv3':
            output = output[:, 1 + getattr(self.model.config, 'num_register_tokens', 0):]
        return output

    def pooled_tokens(self, x, pool_size=3):
        """Keep global prefix tokens and pool only spatial grids within each time slice."""
        from policy.vla_common.tactile_pooling import pool_spatial_tokens, tactile_token_grid
        prefix = 0
        if self.method == 'AnyTouch2':
            output = self.model(self.preprocess(x))
            # Preserve CLS; sensor conditioning tokens stay internal to the backbone.
            tokens = torch.cat((output[:, :1], output[:, 6:]), dim=1)
            prefix = 1
        elif self.mode == 'tactile':
            images = self.preprocess(x).flip(1)
            images = images.transpose(1, 2) if self.method == 'V-JEPA' else images.flatten(1, 2)
            output = self.model.forward_features(images)
            registers = output['x_norm_regtokens']
            prefix = registers.shape[1]
            tokens = torch.cat((registers, output['x_norm_patchtokens']), dim=1)
        else:
            tokens = self.patch_tokens(x)
        grid = tactile_token_grid(self, self.method, self.mode, tokens.shape[1] - prefix)
        return pool_spatial_tokens(tokens, grid, prefix_tokens=prefix, pool_size=pool_size)

    def forward(self, x):
        return self.patch_tokens(x).mean(1)


def initial_metadata(mode, method, size, input_mode):
    """Export a policy-loadable initializer under the representation-specific run path."""
    from datetime import datetime
    from encoder.pretrained_registry import result_root
    from encoder.linear_depth_deform import DEPTH_DEFORM_SPECS, depth_deform_spec
    model = ExternalEncoder(mode, method, size)
    state = dict(model.metadata, initialization='official', input_mode=input_mode, encoder=model.state_dict())
    if input_mode in DEPTH_DEFORM_SPECS:
        state['depth_deform_encoding'] = depth_deform_spec(input_mode)
    out = result_root(input_mode, mode, method, size, 'official') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    from encoder.checkpoint_io import save_checkpoint
    save_checkpoint(state, out / 'encoder.pth')
    (out / 'config.json').write_text(json.dumps({k: v for k, v in state.items() if k != 'encoder'}, indent=2))
    return out / 'encoder.pth'
