"""Standard video-MAE adaptation of the official AnyTouch2 encoder.

The decoder is trained from scratch. This is a single pixel-reconstruction
objective, not a reproduction of AnyTouch2's multi-task pretraining recipe.
"""
import math

import torch
from torch import nn

RECIPE = 'video_mae_masked_pixels_v1'


def patchify_video(images, tubelet=2, patch=16):
    """[B,T,C,H,W] -> time-major [B,L,tubelet*patch*patch*C]."""
    b, t, c, h, w = images.shape
    if t % tubelet or h % patch or w % patch:
        raise ValueError('Video dimensions must be divisible by tubelet/patch sizes')
    return (images.reshape(b, t//tubelet, tubelet, c, h//patch, patch, w//patch, patch)
            .permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(b, -1, tubelet*patch*patch*c))


def tube_mask(batch, times, height, width, ratio, device):
    """Use exactly the same shuffled visible spatial patches at every time."""
    spatial = height * width
    keep = int(spatial * (1-ratio))
    if not 0 < keep < spatial:
        raise ValueError('Tube mask must leave both visible and masked spatial patches')
    chosen = torch.rand(batch, spatial, device=device).argsort(dim=1)[:, :keep]
    ids = (chosen[:, None] + torch.arange(times, device=device)[None, :, None]*spatial).flatten(1)
    mask = torch.ones(batch, times*spatial, dtype=torch.bool, device=device)
    mask.scatter_(1, ids, False)
    return ids, mask


def masked_pixel_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError('Video-MAE prediction, target and mask shapes must match')
    error = (prediction.float()-target.float()).square().mean(-1)
    return error.masked_select(mask).mean()


class VideoMAETraining(nn.Module):
    recipe = RECIPE

    def __init__(self, encoder, mask_ratio=.5, decoder_dim=512, decoder_layers=6, decoder_heads=8):
        super().__init__()
        from transformers import CLIPVisionConfig
        from transformers.models.clip.modeling_clip import CLIPEncoderLayer
        self.encoder, self.mask_ratio = encoder, mask_ratio
        patch = encoder.model.touch_model.config.patch_size
        self.patch, self.tubelet = patch, 2
        h, w = encoder.hw
        self.grid = (encoder.num_frames//self.tubelet, h//patch, w//patch)
        count = math.prod(self.grid)
        self.decoder_embed = nn.Linear(encoder.embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        position = torch.arange(count, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, decoder_dim, 2, dtype=torch.float32)*(-math.log(10000)/decoder_dim))
        encoding = torch.zeros(1, count, decoder_dim)
        encoding[0, :, 0::2], encoding[0, :, 1::2] = torch.sin(position*frequencies), torch.cos(position*frequencies)
        self.register_buffer('decoder_position', encoding)
        config = CLIPVisionConfig(hidden_size=decoder_dim, intermediate_size=4*decoder_dim,
                                  num_attention_heads=decoder_heads, hidden_act='gelu', attention_dropout=0.)
        config._attn_implementation = 'sdpa'
        self.decoder = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(decoder_layers)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.tubelet*patch*patch*3)

    def update_teacher(self, *args, **kwargs):
        # Common training-loop hook; video MAE has no teacher.
        pass

    def _forward(self, images):
        x = self.encoder.preprocess(images)
        ids, mask = tube_mask(len(x), *self.grid, self.mask_ratio, x.device)
        # Positions are added before gathering; masked pixels never enter attention.
        visible = self.encoder.model(x, patch_indices=ids)[:, 6:]
        visible = self.decoder_embed(visible)
        decoded = self.mask_token.to(visible.dtype).expand(len(x), mask.shape[1], -1).clone()
        decoded.scatter_(1, ids[..., None].expand_as(visible), visible)
        decoded = decoded + self.decoder_position.to(decoded.dtype)
        for block in self.decoder:
            output = block(decoded, attention_mask=None, causal_attention_mask=None)
            decoded = output[0] if isinstance(output, tuple) else output
        prediction = self.decoder_pred(self.decoder_norm(decoded))
        # VideoMAE normalizes raw RGB pixels within each tubelet, per channel.
        from encoder.external import resize_frames
        target = patchify_video(resize_frames(images, self.encoder.hw), self.tubelet, self.patch).float()
        target = target.reshape(len(x), -1, self.tubelet*self.patch*self.patch, 3)
        target = (target-target.mean(-2, keepdim=True)) / (target.var(-2, keepdim=True).sqrt()+1e-6)
        target = target.flatten(-2)
        loss = masked_pixel_loss(prediction, target, mask)
        return loss, dict(reconstruction=loss.detach(), mask_fraction=mask.float().mean())

    def forward(self, images):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            loss, metrics = self._forward(images)
        return loss, dict(metrics, _sample_count=loss.new_tensor(len(images)))
