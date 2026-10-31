"""Four-frame MAE with spatial tube masks and compressed clip reconstruction."""
import torch
from torch import nn
from encoder.temporal import TemporalViT as TactileViT
from encoder.vit_common import decoder
from encoder.temporal import patchify, TimeExpansion
from encoder.reconstruction_loss import reconstruction_mse

class MAE(nn.Module):
    def __init__(self, size, in_chans=3, mask_ratio=.75, foreground_weight=50., temporal_compression='4in2'):
        super().__init__()
        self.encoder = TactileViT(size, in_chans=in_chans, temporal_compression=temporal_compression)
        self.mask_ratio = mask_ratio
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.
        d = self.encoder.embed_dim
        n = self.encoder.num_patches
        self.time_expansion = TimeExpansion(256, temporal_compression)
        self.to_decoder = nn.Linear(d, 256)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.pos = nn.Parameter(torch.zeros(1, n, 256))
        self.decode = decoder(256)
        self.pixels = nn.Linear(256, 16 * 16 * in_chans)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, images):
        b, n = images.shape[0], self.encoder.num_patches
        order = torch.rand(b, n, device=images.device).argsort(dim=1)
        visible = order[:, :max(1, int(n * (1 - self.mask_ratio)))]
        encoded = self.encoder.tokens(images, visible)[:, 1:]
        k = self.encoder.output_frames
        visible = (visible[:, None] + torch.arange(k, device=images.device)[None, :, None] * n).flatten(1)
        n *= k
        projected = self.to_decoder(encoded)
        full = self.mask_token.to(projected.dtype).expand(b, n, -1).clone()
        full.scatter_(1, visible.unsqueeze(-1).expand(-1, -1, 256), projected)
        prediction = self.pixels(self.decode(self.time_expansion(full + self.pos.repeat(1, k, 1))))
        target = patchify(images)
        mask = torch.ones(b, n, device=images.device)
        mask.scatter_(1, visible, 0)
        loss = reconstruction_mse(prediction, target, self.foreground_weight,
                                  mask.reshape(b, k, -1).repeat_interleave(4 // k, dim=1).flatten(0, 1).unsqueeze(-1))
        return loss, {'masked_reconstruction': loss.detach()}


Encoder = TactileViT
Model = MAE
