"""Deterministic image -> 196 ViT patch tokens -> image autoencoder."""
from torch import nn
from vit_encoder import TactileViT
from vit_common import PixelDecoder, patchify
from reconstruction_loss import reconstruction_mse


class RAE(nn.Module):
    def __init__(self, size, in_chans=3, foreground_weight=50.):
        super().__init__()
        self.encoder = TactileViT(size, in_chans=in_chans)
        self.decoder = PixelDecoder(self.encoder.embed_dim, in_chans)
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.

    def forward(self, images):
        patches = self.decoder(self.encoder.patch_tokens(images))
        loss = reconstruction_mse(patches, patchify(images), self.foreground_weight)
        return loss, {'reconstruction': loss.detach()}
