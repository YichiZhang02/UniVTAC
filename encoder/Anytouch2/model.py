"""Fixed upstream Anytouch2 architecture and weight loading."""
import json
from pathlib import Path
import torch
from torch import nn

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

    def forward(self, x, patch_indices=None):
        emb = self.touch_model.embeddings
        patches = emb.patch_embedding(x.transpose(1, 2)).flatten(2).transpose(1, 2)
        pos = emb.position_embedding(self.new_position_ids)
        cls = (emb.class_embedding + pos[:, 0]).unsqueeze(1).expand(x.shape[0], -1, -1)
        # UniVTAC optical sensor is GelSight Mini (official sensor ID 3).
        sensor = self.sensor_token[3][None].expand(x.shape[0], -1, -1)
        patches = patches + pos[:, 1:]
        if patch_indices is not None:
            patches = patches.gather(1, patch_indices[..., None].expand(-1, -1, patches.shape[-1]))
        hidden = torch.cat((cls, sensor, patches), 1)
        hidden = self.touch_model.pre_layrnorm(hidden)
        hidden = self.touch_model.encoder(inputs_embeds=hidden).last_hidden_state
        return self.touch_projection(self.touch_model.post_layernorm(hidden))


def build(self, folder, meta, metadata, pretrained, with_decoder, method, size):
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
