"""Fixed official Sparsh MAE architecture."""
from encoder.sparsh_backbone import build as _build

def build(self, folder, meta, metadata, pretrained, with_decoder, method, size):
    return _build(self, folder, meta, metadata, pretrained, with_decoder, 'MAE', size)
