# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Literal
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython

e = IPython.embed


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                             unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models,
                           name)(replace_stride_with_dilation=[False, False, dilation],
                                 weights=torchvision.models.ResNet18_Weights.DEFAULT if name == 'resnet18' else None,
                                 norm_layer=FrozenBatchNorm2d)  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_vision_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class TactileBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, ckpt: str, tac_names:list[str], train_backbone: bool, return_interm_layers: bool, position_embedding, tactile_type:Literal['cls', 'full']='cls'):
        super().__init__()
        
        self.train_backbone = train_backbone
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}

        from .network import Tactile

        self.tac_names = tac_names
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        backbone = Tactile(backbone=name, supervise=[], latent_dims=self.num_channels)
        if ckpt:
            if not Path(ckpt).is_file():
                raise FileNotFoundError(f'Tactile encoder checkpoint not found: {ckpt}')
            state = torch.load(ckpt, map_location='cpu', weights_only=True)
            encoder_state = {key.removeprefix('backbone.'): value for key, value in state.items()
                             if key.startswith('backbone.')}
            backbone.backbone.load_state_dict(encoder_state, strict=True)
            print(f'Loaded tactile encoder backbone from {ckpt}')

        self.tactile_type = tactile_type
        if self.tactile_type == 'cls':
            self.backbone = backbone.backbone
            self.position_embedding = nn.Embedding(1, 512)
        else:
            self.backbone = IntermediateLayerGetter(backbone.backbone, return_layers=return_layers)
            self.position_embedding = position_embedding
 
    def forward(self, x):
        feat, pos = [], []
        if self.tactile_type == 'full':
            xs = self.backbone(x) # dict of feature maps [N, 512, 8, 8]
            for name, x in xs.items():
                feat.append(x)
                pos.append(self.position_embedding(x).to(x.dtype))
        else:
            xs = self.backbone(x) # [N, 512]
            feat.append(xs.unsqueeze(2).unsqueeze(3)) # [N, 512, 1, 1]
            pos.append(self.position_embedding.weight.unsqueeze(-1).unsqueeze(-1)) # [1, 512, 1, 1]
        return feat, pos


class ViTTactileBackbone(nn.Module):
    """Load a tactile ViT checkpoint into ACT."""

    def __init__(self, method, size, ckpt, position_embedding, tactile_type='cls', input_mode='marker_rgb', state=None):
        super().__init__()
        if state is None and (not ckpt or not Path(ckpt).is_file()):
            raise FileNotFoundError(f'ViT tactile checkpoint not found: {ckpt}')
        if method not in ('VAE', 'MAE', 'DINOv2', 'I-JEPA', 'V-JEPA'):
            raise ValueError(f'Unknown tactile encoder method: {method}')
        encoder_dir = Path(__file__).resolve().parents[4] / 'encoder'
        if str(encoder_dir) not in sys.path:
            sys.path.insert(0, str(encoder_dir))
        from vit_encoder import TactileViT
        from VAE.model import VAEEncoder
        from importlib import import_module
        IJEPAEncoder = import_module('I-JEPA.model').IJEPAEncoder
        VJEPAEncoder = import_module('V-JEPA.model').VJEPAEncoder

        if state is None:
            state = torch.load(ckpt, map_location='cpu', weights_only=True)
        if state.get('method') != method or state.get('size') != size:
            raise ValueError(f'Checkpoint is {state.get("method")}/ViT-{state.get("size")}, expected {method}/ViT-{size}')
        if state.get('input_mode', 'marker_rgb') != input_mode:
            raise ValueError(f'Checkpoint input is {state.get("input_mode", "marker_rgb")}, expected {input_mode}')
        if method == 'VAE' and 'encoder' in state and not any(key.startswith('downsample.') for key in state['encoder']):
            raise ValueError('VAE checkpoint uses the old average-pooling encoder; retrain VAE with learned downsampling')
        channels = state.get('in_chans', 3)
        if method == 'VAE':
            self.backbone = VAEEncoder(size, in_chans=channels)
        elif method == 'I-JEPA':
            self.backbone = IJEPAEncoder(size, in_chans=channels)
        elif method == 'V-JEPA':
            self.backbone = VJEPAEncoder(size, in_chans=channels)
        else:
            self.backbone = TactileViT(size, image_size=state['image_size'], in_chans=channels)
        if 'encoder' in state:
            self.backbone.load_state_dict(state['encoder'], strict=True)
        self.num_channels = self.backbone.embed_dim
        self.method = method
        self.tactile_type = tactile_type
        self.position_embedding = position_embedding if tactile_type == 'full' else nn.Embedding(1, 512)
        if tactile_type == 'full' and method in ('MAE', 'DINOv2'):
            self.cls_position = nn.Embedding(1, 512)
        print(f'Loaded {method} ViT-{size} tactile encoder from {ckpt}')

    def forward(self, x):
        if self.tactile_type == 'full':
            if self.method == 'VAE':
                tokens = self.backbone.latent_tokens(x)
                height, width = 7, 7
            elif self.method in ('MAE', 'DINOv2'):
                tokens = self.backbone.tokens(x)
                feat = tokens.transpose(1, 2).unsqueeze(2)
                patches = tokens[:, 1:].transpose(1, 2).reshape(x.shape[0], self.num_channels, 14, 14)
                patch_pos = self.position_embedding(patches).flatten(2).unsqueeze(2)
                cls_pos = self.cls_position.weight[None].transpose(1, 2).unsqueeze(2).expand(x.shape[0], -1, -1, -1)
                pos = torch.cat((cls_pos, patch_pos), dim=-1).to(feat.dtype)
                return [feat], [pos]
            else:
                tokens = self.backbone.patch_tokens(x)
                height, width = 14, 14
            feat = tokens.transpose(1, 2).reshape(x.shape[0], self.num_channels, height, width)
            pos = self.position_embedding(feat).to(feat.dtype)
            return [feat], [pos]
        feat = self.backbone(x)[:, :, None, None]
        pos = self.position_embedding.weight[:, :, None, None]
        return [feat], [pos]


class CNNTactileBackbone(nn.Module):
    """Load a ResNet-S/B reconstruction encoder into ACT."""

    def __init__(self, size, ckpt, state, position_embedding,
                 tactile_type='cls', input_mode='marker_rgb'):
        super().__init__()
        if state.get('method') != 'ResNet' or state.get('size') != size:
            raise ValueError(f'Expected ResNet-{size} checkpoint, got {state.get("method")}-{state.get("size")}')
        if state.get('input_mode') != input_mode:
            raise ValueError(f'Checkpoint input is {state.get("input_mode")}, expected {input_mode}')
        encoder_dir = Path(__file__).resolve().parents[4] / 'encoder'
        if str(encoder_dir) not in sys.path:
            sys.path.insert(0, str(encoder_dir))
        from ResNet.model import ResNetCNNEncoder
        encoder = ResNetCNNEncoder(size, state['in_chans'])
        if 'encoder' in state:
            encoder.load_state_dict(state['encoder'], strict=True)
        self.tactile_type = tactile_type
        self.num_channels = 512 if tactile_type == 'cls' or size == 'S' else 2048
        if tactile_type == 'full':
            self.backbone = IntermediateLayerGetter(encoder.backbone, {'layer4': '0'})
            self.position_embedding = position_embedding
        else:
            self.backbone = encoder
            self.position_embedding = nn.Embedding(1, 512)
        print(f'Loaded ResNet-{size} tactile encoder from {ckpt}')

    def forward(self, x):
        if self.tactile_type == 'full':
            feat = self.backbone(x)['0']
            return [feat], [self.position_embedding(feat).to(feat.dtype)]
        feat = self.backbone(x)[:, :, None, None]
        pos = self.position_embedding.weight[:, :, None, None]
        return [feat], [pos]

def build_tactile_backbone(args):
    train_backbone = args.lr_tactile_backbone > 0
    return_interm_layers = args.tactile_masks
    tactile_type = args.tactile_type if hasattr(args, 'tactile_type') else 'cls'
    position_embedding = build_position_encoding(args)
    method = getattr(args, 'tactile_encoder_method', 'ResNet')
    state = getattr(args, 'tactile_encoder_config', None)
    if state is None and method == 'ResNet' and args.tactile_ckpt:
        state = torch.load(args.tactile_ckpt, map_location='cpu', weights_only=True)
    legacy_resnet = method == 'ResNet' and (state is None or ('encoder' not in state and 'method' not in state))
    if legacy_resnet:
        backbone = TactileBackbone(args.tactile_backbone, args.tactile_ckpt, args.tactile_names,
                                   train_backbone, return_interm_layers, position_embedding, tactile_type)
    elif method == 'ResNet' and state.get('architecture') == 'resnet':
        backbone = CNNTactileBackbone(getattr(args, 'tactile_encoder_size', 'S'), args.tactile_ckpt,
                                      state, position_embedding, tactile_type,
                                      getattr(args, 'tactile_input_mode', 'marker_rgb'))
    else:
        backbone = ViTTactileBackbone(method, getattr(args, 'tactile_encoder_size', 'S'),
                                      args.tactile_ckpt, position_embedding, tactile_type,
                                      getattr(args, 'tactile_input_mode', 'marker_rgb'), state)
    return backbone
