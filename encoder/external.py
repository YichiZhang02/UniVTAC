"""Differentiable external encoders. Constructor metadata supports offline policy restore."""
import importlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from encoder.pretrained_registry import ROOT, SOURCES, source_dir, temporal_spec, validate_identity


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


class AnyTouchBackbone(nn.Module):
    """Official AnyTouch2 encoder, expressed with the installed Transformers CLIP blocks."""
    def __init__(self, config):
        from transformers import CLIPVisionConfig, CLIPVisionModel
        self_config = CLIPVisionConfig(**config['vision_config'])
        super().__init__()
        self.touch_model = CLIPVisionModel(self_config).vision_model
        self.touch_model.embeddings.patch_embedding = nn.Conv3d(3, self_config.hidden_size,
            (2, self_config.patch_size, self_config.patch_size), (2, self_config.patch_size, self_config.patch_size), bias=False)
        n = 2 * (self_config.image_size // self_config.patch_size) ** 2
        self.touch_model.embeddings.position_embedding = nn.Embedding(n + 1, self_config.hidden_size)
        self.sensor_token = nn.Parameter(torch.zeros(20, 5, self_config.hidden_size))
        self.new_position_ids = nn.Parameter(torch.arange(n + 1)[None], requires_grad=False)
        self.touch_projection = nn.Linear(self_config.hidden_size, config['projection_dim'], bias=False)

    def forward(self, x):
        emb = self.touch_model.embeddings
        patches = emb.patch_embedding(x.transpose(1, 2)).flatten(2).transpose(1, 2)
        pos = emb.position_embedding(self.new_position_ids)
        cls = (emb.class_embedding + pos[:, 0]).unsqueeze(1).expand(x.shape[0], -1, -1)
        # UniVTAC optical sensor is GelSight Mini (official sensor ID 3).
        sensor = self.sensor_token[3][None].expand(x.shape[0], -1, -1)
        hidden = torch.cat((cls, sensor, patches + pos[:, 1:]), 1)
        hidden = self.touch_model.pre_layrnorm(hidden)
        hidden = self.touch_model.encoder(inputs_embeds=hidden).last_hidden_state
        return self.touch_projection(self.touch_model.post_layernorm(hidden))


class ExternalEncoder(nn.Module):
    def __init__(self, mode, method, size, *, metadata=None, pretrained=True, with_decoder=False):
        super().__init__()
        validate_identity(mode, method, size)
        self.mode, self.method, self.size = mode, method, size
        self.num_frames, self.frame_stride = temporal_spec(mode, method)
        folder = source_dir(mode, method)
        # A checkpoint may also contain tensors and optimizer state: constructor
        # metadata must stay small, JSON-serializable and independent of those weights.
        meta = {key: value for key, value in (metadata or {}).items()
                if key in ('model_config', 'source_repo', 'source_revision', 'format_version', 'wan_patch_size',
                           'initialization', 'latent_calibration')}
        self.mean, self.std = [0., 0., 0.], [1., 1., 1.]
        if mode == 'tactile' and method != 'AnyTouch2':
            sys.path.insert(0, str(ROOT / 'resources/third_party/sparsh'))
            module = importlib.import_module('tactile_ssl.model.vision_transformer')
            self.embed_dim = 384 if size == 'S' else 768
            # New exports use square inputs; restore the resolution saved by older policies.
            image_size = (metadata or {}).get('image_size', (224, 224))
            self.hw = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
            self.model = module.VisionTransformer(img_size=self.hw, in_chans=3 if method == 'V-JEPA' else 12,
                embed_dim=self.embed_dim, depth=12, num_heads=6 if size == 'S' else 12,
                pos_embed_fn='sinusoidal', num_register_tokens=0 if method == 'MAE' else 1,
                num_frames=4 if method == 'V-JEPA' else 1, tubelet_size=2)
            if pretrained:
                from safetensors.torch import load_file
                paths = list(folder.glob('*.safetensors'))
                if len(paths) != 1:
                    raise FileNotFoundError(f'Expected one official Sparsh weight file in {folder}')
                weights = load_file(paths[0])
                key = 'patch_embed.proj.weight'
                if method != 'V-JEPA':
                    # Preserve original 2-frame response on repeated pairs; all 4 frames participate.
                    weights[key] = weights[key].repeat(1, 2, 1, 1) / 2
                # DINO's learned corruption token belongs to pretraining, not inference.
                weights.pop('masktoken', None)
                self.model.load_state_dict(weights, strict=True)
        elif method == 'AnyTouch2':
            config = meta.get('model_config')
            if config is None:
                config = json.loads((Path(__file__).parent / 'anytouch2_config.json').read_text())
            self.model = AnyTouchBackbone(config)
            self.embed_dim, self.hw = config['projection_dim'], (224, 224)
            self.mean = [.48145466, .4578275, .40821073]
            self.std = [.26862954, .26130258, .27577711]
            if pretrained:
                checkpoint = torch.load(folder / 'checkpoint-4frames.pth', map_location='cpu', weights_only=True)
                checkpoint = checkpoint.get('model', checkpoint)
                weights = {k.split('touch_mae_model.', 1)[1]: v for k, v in checkpoint.items()
                           if 'touch_mae_model.' in k and 'decoder' not in k and 'mask_token' not in k}
                self.model.load_state_dict(weights, strict=True)
            meta['model_config'] = config
        elif method == 'Wan2.2-VAE':
            # Historical policy checkpoints retain their four-frame/48-D contract.
            self.num_frames = (metadata or {}).get('num_frames', self.num_frames)
            self.wan_patch_size = meta.get('wan_patch_size', 1 if self.num_frames == 4 else 2)
            if self.num_frames not in (4, 5) or self.wan_patch_size not in (1, 2):
                raise ValueError('Unsupported Wan temporal/patch contract')
            meta['wan_patch_size'] = self.wan_patch_size
            from diffusers import AutoencoderKLWan
            if pretrained:
                model = AutoencoderKLWan.from_pretrained(folder / 'vae', local_files_only=True, torch_dtype=torch.float32)
            else:
                config = meta.get('model_config')
                if config is None:
                    config = dict(AutoencoderKLWan.load_config(folder / 'vae', local_files_only=True))
                    # Upstream video statistics are not valid for random weights.
                    config['latents_mean'] = [0.] * config['z_dim']
                    config['latents_std'] = [1.] * config['z_dim']
                model = AutoencoderKLWan.from_config(config)
            meta['model_config'] = json.loads(json.dumps(dict(model.config), default=str))
            # Only encoder weights belong to the policy. Decoder is owned by adaptation objective.
            self.model = model
            self.decoder = model.decoder
            self.post_quant_conv = model.post_quant_conv
            model.decoder = None
            model.post_quant_conv = None
            model._cached_conv_counts['decoder'] = 0
            # Keep training-only components out of encoder state_dict and policy parameters.
            decoder, post = self.decoder, self.post_quant_conv
            del self.decoder, self.post_quant_conv
            if pretrained or with_decoder:
                object.__setattr__(self, '_initial_decoder', decoder)
                object.__setattr__(self, '_initial_post_quant_conv', post)
            self.embed_dim, self.hw = model.config.z_dim * self.wan_patch_size**2, (224, 224)
            self.mean, self.std = [.5]*3, [.5]*3
        else:
            from transformers import AutoConfig, AutoModel, SiglipVisionConfig, SiglipVisionModel
            config_dict = meta.get('model_config')
            if method == 'SigLIP2':
                # Fixed-resolution SigLIP2 checkpoints use the Siglip architecture;
                # the separate Siglip2 classes are for NaFlex checkpoints.
                config = (SiglipVisionConfig(**config_dict) if config_dict else
                          AutoConfig.from_pretrained(folder, local_files_only=True).vision_config)
                self.model = (SiglipVisionModel.from_pretrained(folder, local_files_only=True) if pretrained
                              else SiglipVisionModel(config))
                self.hw = (224, 224)
                self.mean, self.std = [.5]*3, [.5]*3
            else:
                config = (AutoConfig.for_model(**config_dict) if config_dict else
                          AutoConfig.from_pretrained(folder, local_files_only=True))
                self.model = (AutoModel.from_pretrained(folder, local_files_only=True) if pretrained
                              else AutoModel.from_config(config))
                if method == 'V-JEPA2':
                    self.model.predictor = None
                self.hw = (256, 256) if method == 'V-JEPA2' else (224, 224)
                self.mean, self.std = [.485, .456, .406], [.229, .224, .225]
            self.embed_dim = config.hidden_size
            meta['model_config'] = config.to_dict()
        self.metadata = dict(meta, pretrained_mode=mode, method=method, size=size,
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
