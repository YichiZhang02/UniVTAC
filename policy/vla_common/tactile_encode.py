"""Adapter from UniVTAC encoder checkpoints to VLA tactile tokens."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn


class TactileEncoder(nn.Module):
    def __init__(self, config, output_dim: int):
        super().__init__()
        path = Path(config.tactile_encoder_path or "")
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=True)
        elif config.tactile_encoder_config:
            state = dict(config.tactile_encoder_config)
        else:
            raise FileNotFoundError(f"UniVTAC tactile encoder checkpoint missing: {path}")
        method = state.get("method")
        size = state.get("size")
        if method not in {"ResNet", "VAE", "MAE", "DINOv2", "I-JEPA", "V-JEPA"}:
            raise ValueError(f"Unsupported UniVTAC tactile checkpoint method: {method}")
        if state.get("input_mode") != config.tactile_input_mode:
            raise ValueError(f"Checkpoint input {state.get('input_mode')} != {config.tactile_input_mode}")
        encoder_dir = Path(__file__).resolve().parents[2] / "encoder"
        if str(encoder_dir) not in sys.path:
            sys.path.insert(0, str(encoder_dir))
        if method == "ResNet":
            from ResNet.model import ResNetCNNEncoder
            self.backbone = ResNetCNNEncoder(size, state["in_chans"])
        elif method == "VAE":
            from VAE.model import VAEEncoder
            self.backbone = VAEEncoder(size, in_chans=state["in_chans"])
        elif method == "I-JEPA":
            self.backbone = importlib.import_module("I-JEPA.model").IJEPAEncoder(
                size, image_size=state["image_size"], in_chans=state["in_chans"]
            )
        elif method == "V-JEPA":
            self.backbone = importlib.import_module("V-JEPA.model").VJEPAEncoder(
                size, in_chans=state["in_chans"]
            )
        else:
            from vit_encoder import TactileViT
            self.backbone = TactileViT(size, image_size=state["image_size"], in_chans=state["in_chans"])
        if "encoder" in state:
            self.backbone.load_state_dict(state["encoder"], strict=True)
        self.method = method
        self.mode = config.tactile_type
        self.keys = list(config.tactile_encoder_keys())
        self.image_size = int(state.get("image_size", 224))
        if not self.keys:
            raise ValueError("encode mode requires tactile keys")
        if config.freeze_tactile_encoder:
            self.backbone.requires_grad_(False)
        feature_dim = 2048 if method == "ResNet" and size == "B" and self.mode == "full" else self.backbone.embed_dim
        self.proj = nn.Linear(feature_dim, output_dim)
        config.tactile_encoder_config = {
            "method": method, "size": size, "input_mode": state["input_mode"],
            "image_size": self.image_size, "in_chans": state["in_chans"],
            "tactile_type": self.mode,
        }

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if self.mode == "cls":
            return self.backbone(x).unsqueeze(1)
        if self.method == "ResNet":
            net = self.backbone.backbone
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
            return x.flatten(2).transpose(1, 2)
        if self.method == "VAE":
            return self.backbone.latent_tokens(x)
        if self.method in {"MAE", "DINOv2"}:
            return self.backbone.tokens(x)
        return self.backbone.patch_tokens(x)

    def forward_flat(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for key in self.keys:
            if key not in batch:
                raise KeyError(f"Missing tactile input {key}")
            x = batch[key]
            if x.ndim == 5:
                b, f, c, h, w = x.shape
                x = x.reshape(b * f, c, h, w)
                encoded = self._tokens(x).reshape(b, -1, self.proj.in_features)
            elif x.ndim == 4:
                encoded = self._tokens(x)
            else:
                raise ValueError(f"Expected [B,C,H,W] or [B,F,C,H,W], got {tuple(x.shape)}")
            tokens.append(self.proj(encoded.to(self.proj.weight.dtype)))
        return torch.cat(tokens, dim=1)

    forward = forward_flat
