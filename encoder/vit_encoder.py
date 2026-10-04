"""Shared tactile ViT used by pretraining and ACT.

Checkpoints store this module's state under the ``encoder`` key.
"""
import torch
from torch import nn
from torch.nn import functional as F


VIT_NAMES = {'S': 'vit_small_patch16_224', 'B': 'vit_base_patch16_224'}


class TactileViT(nn.Module):
    def __init__(self, size='S', image_size=224, in_chans=3):
        super().__init__()
        if size not in VIT_NAMES:
            raise ValueError(f'Expected ViT size S or B, got {size}')
        import timm
        self.size = size
        self.image_size = image_size
        self.in_chans = in_chans
        self.vit = timm.create_model(VIT_NAMES[size], pretrained=False,
                                     num_classes=0, img_size=image_size, in_chans=in_chans)
        self.embed_dim = self.vit.num_features
        self.grid_size = image_size // 16
        self.num_patches = self.grid_size ** 2

    def tokens(self, images, keep_indices=None, patch_mask=None, mask_token=None):
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(images, size=(self.image_size, self.image_size),
                                   mode='bilinear', align_corners=False)
        x = self.vit.patch_embed(images)
        if patch_mask is not None:
            x = torch.where(patch_mask.unsqueeze(-1), mask_token.to(x.dtype), x)
        pos = self.vit.pos_embed[:, 1:]
        if keep_indices is not None:
            index = keep_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = x.gather(1, index)
            pos = pos.expand(x.shape[0], -1, -1).gather(1, index)
        x = x + pos
        cls = self.vit.cls_token.expand(x.shape[0], -1, -1) + self.vit.pos_embed[:, :1]
        x = self.vit.pos_drop(torch.cat((cls, x), dim=1))
        for block in self.vit.blocks:
            x = block(x)
        return self.vit.norm(x)

    def forward(self, images):
        return self.tokens(images)[:, 0]

    def patch_tokens(self, images):
        return self.tokens(images)[:, 1:]
