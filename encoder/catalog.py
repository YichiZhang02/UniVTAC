"""Public encoder names and initialization/size contracts.

Legacy (mode, method) names remain readable for existing checkpoints. New
commands use explicit Sparsh names and ``official`` as the fixed model size.
"""
SINGLE_FRAME_METHODS = ('ResNet', 'RVAE', 'RAE', 'SD-VAE', 'MAE', 'DINOv2', 'I-JEPA', 'V-JEPA')
MULTIFRAME_METHODS = tuple(name + '4Frame' for name in SINGLE_FRAME_METHODS)
LOCAL_METHODS = SINGLE_FRAME_METHODS + MULTIFRAME_METHODS


def base_method(method):
    name = ALIASES.get(method, method)
    return name[:-6] if name in MULTIFRAME_METHODS else name


EXTERNAL_METHODS = {
    'Sparsh_mae': ('tactile', 'MAE', 'S'),
    'Sparsh_DINOv2': ('tactile', 'DINOv2', 'B'),
    'Sparsh_IJEPA': ('tactile', 'I-JEPA', 'S'),
    'Sparsh_VJEPA': ('tactile', 'V-JEPA', 'S'),
    'Wan22_VAE': ('vision', 'Wan2.2-VAE', 'B'),
    'Anytouch2': ('tactile', 'AnyTouch2', 'B'),
    # Existing vision backends remain available with fixed upstream sizes.
    'DINOv3': ('vision', 'DINOv3', 'S'),
    'SigLIP2': ('vision', 'SigLIP2', 'B'),
    'V-JEPA2': ('vision', 'V-JEPA2', 'L'),
}
ALIASES = {'VAE4Frame': 'RVAE4Frame', 'VAE': 'RVAE', 'IJEPA': 'I-JEPA', 'IJPEA': 'I-JEPA', 'VJEPA': 'V-JEPA',
           'Sparsh_MAE': 'Sparsh_mae', 'AnyTouch2': 'Anytouch2', 'Wan2.2-VAE': 'Wan22_VAE'}


def canonical_name(mode, method):
    if mode != 'univtac':
        for name, (legacy_mode, legacy_method, _) in EXTERNAL_METHODS.items():
            if (mode, method) == (legacy_mode, legacy_method):
                return name
    return ALIASES.get(method, method)


def resolve_identity(mode, method, size=None):
    """Translate a public name into the upstream architecture identity."""
    name = canonical_name(mode, method)
    if name in EXTERNAL_METHODS:
        legacy_mode, legacy_method, width = EXTERNAL_METHODS[name]
        return legacy_mode, legacy_method, width if size in (None, 'official') else size
    return mode, name, size


def selection(mode, method, size=None, initialization=None):
    """Validate a new training selection; external widths cannot be changed."""
    name = canonical_name(mode, method)
    if name in EXTERNAL_METHODS:
        mode = EXTERNAL_METHODS[name][0]
        size = size or 'official'
        if size != 'official':
            raise ValueError(f'{name} has a fixed upstream architecture; size must be official')
        initialization = initialization or 'official'
        allowed = ('scratch', 'official', 'adapted')
    elif name in LOCAL_METHODS:
        if mode != 'univtac':
            raise ValueError(f'{name} is a scratch UniVTAC encoder')
        size = size or 'S'
        if size not in ('S', 'B'):
            raise ValueError(f'{name} supports size S or B')
        initialization = initialization or 'scratch'
        allowed = ('scratch',)
    else:
        raise ValueError(f'Unknown encoder: {method}')
    if initialization not in allowed:
        raise ValueError(f'{name} supports initialization {allowed}, got {initialization}')
    return mode, name, size, initialization
