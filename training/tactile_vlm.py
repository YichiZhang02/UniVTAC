"""Qwen3.5 V→T→L assembly and explicit masks for its full-attention layers.

GatedDeltaNet receives the original 2-D padding mask and stays causal.
Only tactile 3-D coordinates follow UniVTAC1's sensor-local (t, h, w) layout;
no encoder, mask, or other policy settings are imported from that repository.
"""
import argparse
from dataclasses import dataclass

import torch
from torch import nn


FUSION = ('vlm', 'head')
WRAPPERS = ('none', 'vision', 'tactile')
ROPES = ('none', '1d', '3d')
MASKS = ('causal', 'modality', 'bidirectional')


def validate_config(cfg):
    for key, choices, default in (
        ('tactile_fusion', FUSION, 'vlm'),
        ('tactile_wrapper', WRAPPERS, 'none'),
        ('tactile_rope', ROPES, '1d'),
        ('vlm_attention_mask', MASKS, 'causal'),
    ):
        if cfg.get(key, default) not in choices:
            raise ValueError(f'{key} must be one of {choices}')


def make_attention_mask(valid, modalities, mode, dtype):
    """Return additive [B,1,Q,K] mask (0=visible, min-float=blocked)."""
    if mode not in MASKS:
        raise ValueError(mode)
    n = valid.shape[1]
    if mode == 'causal':
        visible = torch.ones(n, n, device=valid.device, dtype=torch.bool).tril()[None]
    elif mode == 'modality':
        # Prefix(-1), V(0), T(1), L(2); each block is bidirectional.
        visible = modalities[:, :, None] >= modalities[:, None, :]
    else:
        visible = torch.ones_like(valid[:, :, None] & valid[:, None, :])
    visible = visible & valid[:, :, None] & valid[:, None, :]
    return torch.zeros(visible.shape, device=valid.device, dtype=dtype).masked_fill(
        ~visible, torch.finfo(dtype).min)[:, None]


def grid_coordinates(frames, height, width, device):
    # UniVTAC1's 3-D convention: chronological time, row, column; each hand
    # has its own local grid and shares the same temporal origin.
    return torch.stack(torch.meshgrid(torch.arange(frames, device=device),
        torch.arange(height, device=device), torch.arange(width, device=device),
        indexing='ij'), -1).reshape(-1, 3)


@dataclass
class PackedTactile:
    embeddings: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    modalities: torch.Tensor
    tactile_span: torch.Tensor
    tactile_body: torch.Tensor
    original_valid: torch.Tensor

    def split(self, hidden):
        b, _, d = hidden.shape
        vl = hidden[~self.tactile_span].reshape(b, -1, d)
        tactile = hidden[self.tactile_body].reshape(b, -1, d)
        return vl, tactile


class TactileVLM(nn.Module):
    """Assembly module; Qwen itself is passed to forward, never registered twice."""
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        validate_config(cfg)
        self.wrapper = cfg.get('tactile_wrapper', 'none')
        self.rope = cfg.get('tactile_rope', '1d')
        self.mask_mode = cfg.get('vlm_attention_mask', 'causal')
        self.delimiters = nn.Embedding(2, hidden_dim) if self.wrapper == 'tactile' else None
        if self.delimiters is not None:
            nn.init.normal_(self.delimiters.weight, std=.02)

    def pack(self, inputs, embeddings, positions, tactile, layout, vision_start_id,
             vision_end_id, token_embedding):
        ids = inputs['input_ids']
        valid = inputs['attention_mask'].bool()
        b, n, d = embeddings.shape
        sensors, frames, height, width = layout
        count = frames * height * width
        if tactile.shape != (b, sensors*count, d):
            raise ValueError('Tactile token count does not match the encoder grid')
        coordinates = grid_coordinates(frames, height, width, ids.device)
        wrapped = self.wrapper != 'none'
        if self.wrapper == 'vision':
            delimiters = token_embedding(torch.tensor([vision_start_id, vision_end_id], device=ids.device))
        elif self.wrapper == 'tactile':
            delimiters = self.delimiters.weight
        else:
            delimiters = None
        pieces, coords, body_flags = [], [], []
        extent = 0
        for sensor in range(sensors):
            body = tactile[:, sensor*count:(sensor+1)*count]
            if wrapped:
                pieces.extend((delimiters[0][None,None].expand(b,1,-1), body,
                               delimiters[1][None,None].expand(b,1,-1)))
                span = max(frames,height,width)
                coords.extend((torch.full((1,3),extent,device=ids.device),
                               coordinates+extent+1,
                               torch.full((1,3),extent+span+1,device=ids.device)))
                body_flags.extend([False]+[True]*count+[False])
                extent += span+2
            else:
                pieces.append(body)
                coords.append(coordinates)
                body_flags.extend([True]*count)
                extent = max(frames,height,width)
        extra = torch.cat(pieces,1).to(embeddings.dtype)
        coordinate = torch.cat(coords).long()
        inserted = extra.shape[1]
        if self.rope == '1d':
            coordinate = torch.arange(inserted,device=ids.device)[:,None].expand(-1,3)
            extent = inserted
        elif self.rope == 'none':
            coordinate = torch.zeros_like(coordinate)
            # No phase and no positional span for tactile content. Delimiters
            # also have identity RoPE in this ablation.
            extent = 0
        out_embeddings, out_valid, out_pos, out_mod, out_span, out_body = [], [], [], [], [], []
        body_flags = torch.tensor(body_flags,device=ids.device,dtype=torch.bool)
        for index in range(b):
            end = torch.where((ids[index] == vision_end_id) & valid[index])[0]
            start = torch.where((ids[index] == vision_start_id) & valid[index])[0]
            if not len(end) or not len(start):
                raise ValueError('V→T→L requires at least one image per sample')
            split = int(end[-1])+1
            base = int(positions[:,index,:split][:,valid[index,:split]].max())+1
            tp = coordinate.T+base if self.rope != 'none' else coordinate.T
            tail_pos = positions[:,index,split:]+extent
            out_embeddings.append(torch.cat((embeddings[index,:split],extra[index],embeddings[index,split:])))
            out_valid.append(torch.cat((valid[index,:split],valid.new_ones(inserted),valid[index,split:])))
            out_pos.append(torch.cat((positions[:,index,:split],tp,tail_pos),dim=1))
            mod = torch.full((n+inserted,),2,device=ids.device,dtype=torch.long)
            mod[:int(start[0])] = -1
            mod[int(start[0]):split] = 0
            mod[split:split+inserted] = 1
            out_mod.append(mod)
            span_mask = valid.new_zeros(n+inserted);span_mask[split:split+inserted] = True
            body_mask = valid.new_zeros(n+inserted);body_mask[split:split+inserted] = body_flags
            out_span.append(span_mask);out_body.append(body_mask)
        return PackedTactile(torch.stack(out_embeddings),torch.stack(out_valid),
            torch.stack(out_pos,dim=1),torch.stack(out_mod),torch.stack(out_span),
            torch.stack(out_body),valid)

    def forward(self, vlm, images, instructions, tactile=None, layout=None):
        inputs = vlm.build_qwenvl_inputs(images, instructions)
        core = vlm.model.model
        text = core.language_model
        if getattr(text.config,'model_type',None) != 'qwen3_5_text':
            raise ValueError('VLM tactile injection currently requires the Qwen3.5 backbone')
        embeddings = core.get_input_embeddings()(inputs['input_ids'])
        image_output = core.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'],return_dict=True)
        image_features = torch.cat(image_output.pooler_output,dim=0).to(embeddings.dtype)
        image_mask, _ = core.get_placeholder_mask(inputs['input_ids'],inputs_embeds=embeddings,
                                                image_features=image_features)
        embeddings = embeddings.masked_scatter(image_mask,image_features)
        positions, _ = core.get_rope_index(inputs['input_ids'],
            image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs['attention_mask'],
            mm_token_type_ids=inputs.get('mm_token_type_ids'))
        if tactile is None:
            valid = inputs['attention_mask'].bool()
            modalities = torch.full_like(inputs['input_ids'], 2)
            for index in range(len(valid)):
                start = torch.where((inputs['input_ids'][index] == vlm.model.config.vision_start_token_id) & valid[index])[0]
                end = torch.where((inputs['input_ids'][index] == vlm.model.config.vision_end_token_id) & valid[index])[0]
                if not len(start) or not len(end):
                    raise ValueError('Modality masks require an image block')
                modalities[index,:int(start[0])] = -1
                modalities[index,int(start[0]):int(end[-1])+1] = 0
            packed = PackedTactile(embeddings,valid,positions,modalities,
                torch.zeros_like(valid),torch.zeros_like(valid),valid)
        else:
            packed = self.pack(inputs,embeddings,positions,tactile,layout,
                vlm.model.config.vision_start_token_id,vlm.model.config.vision_end_token_id,
                core.get_input_embeddings())
        return self.run_text(text,packed), packed

    def run_text(self, text, packed):
        # Calling decoder layers directly avoids feeding a 4-D attention mask
        # to GatedDeltaNet. Its upstream forward and padding mask stay intact.
        hidden = packed.embeddings
        rotary = text.rotary_emb(hidden,packed.positions)
        full_mask = make_attention_mask(packed.valid,packed.modalities,self.mask_mode,hidden.dtype)
        layers = []
        for layer in text.layers:
            mask = packed.valid if layer.layer_type == 'linear_attention' else full_mask
            hidden = layer(hidden,position_embeddings=rotary,attention_mask=mask,
                           past_key_values=None,position_ids=None,use_cache=False)
            layers.append(hidden)
        layers[-1] = text.norm(layers[-1])
        return layers


def add_tactile_arguments(parser):
    parser.add_argument('--freeze-tactile-encoder',action=argparse.BooleanOptionalAction,default=None)
    for flag, choices in (('tactile-fusion',FUSION),('tactile-wrapper',WRAPPERS),
                          ('tactile-rope',ROPES),('vlm-attention-mask',MASKS)):
        parser.add_argument('--'+flag, choices=choices)


def apply_tactile_arguments(args, cfg):
    frozen = getattr(args,'freeze_tactile_encoder',None)
    if frozen is not None:
        cfg['freeze_tactile_encoder'] = frozen
    for key in ('tactile_fusion','tactile_wrapper','tactile_rope','vlm_attention_mask'):
        value = getattr(args,key,None)
        if value is not None:
            cfg[key] = value
    validate_config(cfg)
