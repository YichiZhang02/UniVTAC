"""Construct an exported local encoder without allocating its training heads."""
from importlib import import_module
from encoder.catalog import canonical_name, LOCAL_METHODS


def local_encoder(method, size, in_chans=3, *, num_frames=1, image_size=224):
    method = canonical_name('univtac', method)
    if method not in LOCAL_METHODS:
        raise ValueError(f'Unsupported local encoder: {method}')
    classes = {'ResNet': 'ResNetCNNEncoder', 'RVAE': 'RVAEEncoder',
               'SD-VAE': 'SDVAEEncoder', 'I-JEPA': 'IJEPAEncoder', 'V-JEPA': 'VJEPAEncoder'}
    if method in classes:
        cls = getattr(import_module(f'encoder.{method}.model'), classes[method])
        options = {'num_frames': num_frames} if method == 'V-JEPA' else {}
        return cls(size, in_chans=in_chans, **options)
    from encoder.vit_encoder import TactileViT
    return TactileViT(size, image_size=image_size, in_chans=in_chans)
