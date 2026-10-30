"""Fixed official Sparsh DINOv2 architecture."""
from encoder.sparsh_backbone import build as _build

def build(self, folder, meta, metadata, pretrained, with_decoder, method, size):
    return _build(self, folder, meta, metadata, pretrained, with_decoder, 'DINOv2', size)
