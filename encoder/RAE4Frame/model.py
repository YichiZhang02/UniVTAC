"""Four-frame clip -> one/two compressed spatial grids -> four-frame reconstruction."""
from torch import nn
from encoder.temporal import TemporalViT as TactileViT
from encoder.vit_common import PixelDecoder
from encoder.temporal import patchify, TimeExpansion
from encoder.reconstruction_loss import reconstruction_mse


class RAE(nn.Module):
    def __init__(self, size, in_chans=3, foreground_weight=50., temporal_compression='4in2'):
        super().__init__()
        self.encoder = TactileViT(size, in_chans=in_chans, temporal_compression=temporal_compression)
        self.decoder = PixelDecoder(self.encoder.embed_dim, in_chans)
        self.time_expansion = TimeExpansion(self.encoder.embed_dim, temporal_compression)
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.

    def forward(self, images):
        patches = self.decoder(self.time_expansion(self.encoder.patch_tokens(images)))
        loss = reconstruction_mse(patches, patchify(images), self.foreground_weight)
        return loss, {'reconstruction': loss.detach()}


Encoder = TactileViT
Model = RAE
