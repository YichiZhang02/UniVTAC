"""Fixed official Sparsh V-JEPA architecture."""
from encoder.sparsh_backbone import build as _build

def build(self, folder, meta, metadata, pretrained, with_decoder, method, size):
    return _build(self, folder, meta, metadata, pretrained, with_decoder, 'V-JEPA', size)
