"""Fixed upstream Wan22_VAE architecture and weight loading."""
import json
import torch
from pathlib import Path

def build(self, folder, meta, metadata, pretrained, with_decoder, method, size):
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
            config_folder = folder / 'vae' if (folder / 'vae/config.json').is_file() else Path(__file__).parent
            if config_folder == Path(__file__).parent:
                config = json.loads((config_folder / 'architecture.json').read_text())
            else:
                config = dict(AutoencoderKLWan.load_config(config_folder, local_files_only=True))
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
