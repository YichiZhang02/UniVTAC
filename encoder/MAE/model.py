"""MAE tactile encoder pretraining objective."""
import torch
from torch import nn
from vit_encoder import TactileViT
from vit_common import decoder, patchify
from reconstruction_loss import reconstruction_mse

class MAE(nn.Module):
    def __init__(self, size, in_chans=3, mask_ratio=.75, foreground_weight=50.):
        super().__init__()
        self.encoder = TactileViT(size, in_chans=in_chans)
        self.mask_ratio = mask_ratio
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.
        d = self.encoder.embed_dim
        n = self.encoder.num_patches
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
        projected = self.to_decoder(encoded)
        full = self.mask_token.to(projected.dtype).expand(b, n, -1).clone()
        full.scatter_(1, visible.unsqueeze(-1).expand(-1, -1, 256), projected)
        prediction = self.pixels(self.decode(full + self.pos))
        target = patchify(images)
        mask = torch.ones(b, n, device=images.device)
        mask.scatter_(1, visible, 0)
        loss = reconstruction_mse(prediction, target, self.foreground_weight,
                                  mask.unsqueeze(-1))
        return loss, {'masked_reconstruction': loss.detach()}
