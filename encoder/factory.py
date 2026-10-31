"""Construct an exported local encoder without allocating its training heads."""
from importlib import import_module
from encoder.catalog import canonical_name, LOCAL_METHODS, MULTIFRAME_METHODS


def require_current_local_checkpoint(state):
    """Reject obsolete temporal architectures and unpacked SD-VAE contracts."""
    method = canonical_name('univtac', state['method'])
    if method in MULTIFRAME_METHODS and state.get('temporal_backend') != 'conv3d':
        raise ValueError('4Frame checkpoints require temporal_backend=conv3d; retrain the old encoder')
    if method in ('SD-VAE', 'SD-VAE4Frame') and state.get('sd_patch_size') != 2:
        raise ValueError('SD-VAE checkpoints require sd_patch_size=2; re-export or retrain the encoder')


def local_encoder(method, size, in_chans=3, *, num_frames=1, image_size=224, temporal_compression='4in2', sd_patch_size=2):
    method = canonical_name('univtac', method)
    if method not in LOCAL_METHODS:
        raise ValueError(f'Unsupported local encoder: {method}')
    if method in MULTIFRAME_METHODS:
        if num_frames != 4 or image_size != 224:
            raise ValueError('Multi-frame encoders require num_frames=4 and image_size=224')
        cls = import_module(f'encoder.{method}.model').Encoder
        options = {'sd_patch_size': sd_patch_size} if method == 'SD-VAE4Frame' else {}
        return cls(size, in_chans=in_chans, temporal_compression=temporal_compression, **options)
    classes = {'ResNet': 'ResNetCNNEncoder', 'RVAE': 'RVAEEncoder',
               'SD-VAE': 'SDVAEEncoder', 'I-JEPA': 'IJEPAEncoder', 'V-JEPA': 'VJEPAEncoder'}
    if method in classes:
        cls = getattr(import_module(f'encoder.{method}.model'), classes[method])
        options = {'num_frames': num_frames} if method == 'V-JEPA' else {}
        if method == 'SD-VAE':
            options['sd_patch_size'] = sd_patch_size
        return cls(size, in_chans=in_chans, **options)
    from encoder.vit_encoder import TactileViT
    return TactileViT(size, image_size=image_size, in_chans=in_chans)
