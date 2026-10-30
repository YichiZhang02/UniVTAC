"""Encoder identity is (pretrained_mode, method, size), never method alone."""
from pathlib import Path
import json
from encoder.catalog import LOCAL_METHODS, EXTERNAL_METHODS, canonical_name, resolve_identity

ROOT = Path(__file__).resolve().parents[1]
SPARSH_REVISION = 'fee6a05a97330eca014b59c71ac41596fe3974a7'
MODES = ('univtac', 'tactile', 'vision')
METHODS = {
    'univtac': LOCAL_METHODS + ('VAE',),
    'tactile': ('MAE', 'DINOv2', 'I-JEPA', 'V-JEPA', 'AnyTouch2'),
    'vision': ('DINOv3', 'SigLIP2', 'V-JEPA2', 'Wan2.2-VAE'),
}
# Only files needed by the encoder are downloaded (no video diffusion / text models).
SOURCES = {
    ('tactile', 'MAE'): ('S', 'facebook/sparsh-mae-small', ['*.safetensors', 'README.md']),
    ('tactile', 'DINOv2'): ('B', 'facebook/sparsh-dinov2-base', ['*.safetensors', 'README.md']),
    ('tactile', 'I-JEPA'): ('S', 'facebook/sparsh-ijepa-small', ['*.safetensors', 'README.md']),
    ('tactile', 'V-JEPA'): ('S', 'facebook/sparsh-vjepa-small', ['*.safetensors', 'README.md']),
    ('tactile', 'AnyTouch2'): ('B', 'xxuan01/AnyTouch2-Model', ['checkpoint-4frames.pth', 'README.md']),
    ('vision', 'DINOv3'): ('S', 'facebook/dinov3-vits16-pretrain-lvd1689m', ['*.json', '*.safetensors', 'LICENSE.md']),
    ('vision', 'SigLIP2'): ('B', 'google/siglip2-base-patch16-224', ['config.json', 'preprocessor_config.json', '*.safetensors']),
    ('vision', 'V-JEPA2'): ('L', 'facebook/vjepa2-vitl-fpc64-256', ['config.json', 'video_preprocessor_config.json', 'model.safetensors']),
    ('vision', 'Wan2.2-VAE'): ('B', 'Wan-AI/Wan2.2-TI2V-5B-Diffusers', ['vae/*']),
}


def validate_identity(mode, method, size=None):
    public_name = method
    if public_name in EXTERNAL_METHODS and size not in (None, 'official'):
        raise ValueError(f'{public_name} requires size official')
    mode, method, size = resolve_identity(mode, method, size)
    if mode not in MODES or method not in METHODS[mode]:
        raise ValueError(f'Unsupported encoder {mode}/{method}; choices: {METHODS}')
    if mode == 'univtac' and size is not None and size not in ('S', 'B'):
        raise ValueError(f'UniVTAC encoders support S or B, got {size}')
    if mode != 'univtac' and size is not None and size != SOURCES[mode, method][0]:
        raise ValueError(f'{mode}/{method}: supported downloaded size is {SOURCES[mode, method][0]}, got {size}')


def temporal_spec(mode, method):
    mode, method, _ = resolve_identity(mode, method)
    if mode == 'univtac':
        return (1, 1)
    if method == 'Wan2.2-VAE':
        return (5, 2)
    return (4, 2) if mode == 'tactile' or method in ('V-JEPA', 'AnyTouch2', 'V-JEPA2', 'Wan2.2-VAE') else (1, 1)


def source_dir(mode, method):
    mode, method, _ = resolve_identity(mode, method)
    size = SOURCES[mode, method][0]
    return ROOT / 'resources/pretrained_models' / mode / method / size


def default_size(mode, method):
    validate_identity(mode, method)
    return 'official' if canonical_name(mode, method) in EXTERNAL_METHODS else 'S' if mode == 'univtac' else SOURCES[mode, method][0]


def result_root(input_mode, mode, method, size, initialization=None):
    validate_identity(mode, method, size)
    mode = resolve_identity(mode, method)[0]
    root = ROOT / 'encoder_results' / input_mode / mode / method / size
    if initialization is not None:
        if mode == 'univtac' and initialization != 'scratch':
            raise ValueError('Local encoders only support scratch initialization')
        if initialization not in ('scratch', 'official', 'adapted'):
            raise ValueError('initialization must be scratch, official or adapted')
        root = root / initialization
    return root


def checkpoint_initialization(metadata):
    # Historical local methods were random; historical external runs loaded
    # official weights. New exports always store an explicit provenance field.
    return metadata.get('initialization', 'scratch' if metadata.get('pretrained_mode', 'univtac') == 'univtac' else 'official')


def require_initialization(metadata, requested):
    actual = checkpoint_initialization(metadata)
    if requested is not None and requested != actual:
        raise ValueError(f'Encoder initialization mismatch: requested {requested}, checkpoint is {actual}')
    return actual


def latest_checkpoint(input_mode, mode, method, size, initialization):
    """Select by initialization before mtime; inspect small configs, not weights."""
    root = result_root(input_mode, mode, method, size)
    candidates = []
    roots = [root]
    if mode == 'univtac' and method == 'RVAE':
        roots.append(ROOT / 'encoder_results' / input_mode / mode / 'VAE' / size)
    for path in (p for directory in roots for p in directory.rglob('encoder.pth')):
        config_path = path.parent / 'config.json'
        if config_path.exists():
            metadata = json.loads(config_path.read_text())
        else:
            # New layout is unambiguous even when an export lacks config.json.
            first = path.parent.parent.name if path.parent.parent.name in ('scratch', 'official', 'adapted') else ''
            metadata = {'pretrained_mode': mode}
            if first in ('scratch', 'official', 'adapted'):
                metadata['initialization'] = first
        if checkpoint_initialization(metadata) == initialization:
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def history_indices(step, num_frames=4, stride=2):
    return [max(0, step - offset * stride) for offset in reversed(range(num_frames))]


def constructor_metadata(checkpoint):
    """Only path-free constructor fields belong in a policy's YAML config."""
    keys = ('format_version', 'color_order', 'pretrained_mode', 'method', 'size',
            'architecture', 'image_size', 'input_mode', 'in_chans', 'depth_deform_encoding',
            'num_frames', 'frame_stride', 'background_offset', 'model_config', 'source_repo', 'source_revision',
            'wan_patch_size', 'initialization', 'latent_calibration')
    return {key: checkpoint[key] for key in keys if key in checkpoint}
