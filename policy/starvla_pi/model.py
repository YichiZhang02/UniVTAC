"""UniVTAC adapter for the upstream QwenPI_v3 layer-wise flow head."""
import sys
from pathlib import Path

import torch
from torch import nn


class StarVLAPI(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
        from omegaconf import OmegaConf
        from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
        from training.models import Tactile

        conf = OmegaConf.create({
            'framework': {'name': 'QwenPI_v3', 'qwenvl': {
                'base_vlm': cfg['base_vlm'], 'attn_implementation': 'sdpa'},
                'action_model': {'action_dim': dim, 'state_dim': dim,
                    'action_horizon': cfg['chunk'],
                    'repeated_diffusion_steps': cfg.get('flow_repeats', 8),
                    'num_target_vision_tokens': 0,
                    'num_inference_timesteps': cfg.get('num_inference_timesteps', 10),
                    'diffusion_model_cfg': {'action_dit_hidden_dim': 1024,
                        'joint_action_attention': True, 'interleave_self_attention': False}}},
            'trainer': {'repeated_diffusion_steps': cfg.get('flow_repeats', 8)},
            'datasets': {'vla_data': {}}})
        self.native = Qwen_PI_v3(conf)
        self.native.qwen_vl_interface.model.gradient_checkpointing_disable()
        self.native.qwen_vl_interface.model.config.use_cache = False
        vlm_config = self.native.qwen_vl_interface.model.config
        getattr(vlm_config, 'text_config', vlm_config).use_cache = False
        self.repeats = cfg.get('flow_repeats', 8)
        self.tactile = Tactile(cfg) if cfg['tactile'] else None
        from training.tactile_vlm import TactileVLM, validate_config
        validate_config(cfg)
        self.fusion_mode = cfg.get('tactile_fusion', 'vlm')
        hidden = self.native.config.framework.qwenvl.vl_hidden_dim
        inject_vlm = self.tactile is not None and self.fusion_mode != 'head'
        self.project = nn.Linear(self.tactile.dim, hidden if inject_vlm else 1024) if self.tactile else None
        self.inject_tactile_vlm = inject_vlm
        self.vlm_tactile = TactileVLM(cfg if inject_vlm else dict(cfg,tactile_wrapper='none'), hidden) \
            if inject_vlm or cfg.get('vlm_attention_mask','causal') != 'causal' else None

    def encode(self, batch):
        from PIL import Image
        images = [[Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                   for x in sample] for sample in batch['image']]
        instructions = self.native.add_discretized_state_to_instruction(
            batch['lang'], batch['state'][:, None].detach().float().cpu().numpy())
        if self.inject_tactile_vlm:
            tactile = self.project(self.tactile(batch['tactile']))
            layers, packed = self.vlm_tactile(self.native.qwen_vl_interface, images,
                instructions, tactile, self.tactile.token_layout(batch['tactile']))
            return self.native._project_vl_hidden_for_action(layers), packed.valid
        if self.vlm_tactile is not None:
            layers, packed = self.vlm_tactile(self.native.qwen_vl_interface, images, instructions)
            hidden, mask = self.native._project_vl_hidden_for_action(layers), packed.valid
        else:
            hidden, mask = self.native._encode_vl_hidden_states(images, instructions)
        if mask is not None:
            mask = mask.bool()
        if self.tactile:
            tokens = self.project(self.tactile(batch['tactile'])).to(hidden[-1].dtype)
            hidden = [torch.cat((h, tokens), 1) for h in hidden]
            if mask is not None:
                mask = torch.cat((mask, torch.ones(tokens.shape[:2], device=mask.device,
                                                  dtype=torch.bool)), 1)
        return hidden, mask

    def forward(self, batch):
        hidden, mask = self.encode(batch)
        return self.native.action_model(
            [h.repeat(self.repeats, 1, 1) for h in hidden],
            batch['action'].to(hidden[-1].dtype).repeat(self.repeats, 1, 1),
            None, encoder_attention_mask=mask.repeat(self.repeats, 1) if mask is not None else None)

    @torch.inference_mode()
    def predict_action(self, batch):
        hidden, mask = self.encode(batch)
        return self.native.action_model.predict_action(hidden, None, encoder_attention_mask=mask)
