"""Shared official Sparsh architecture construction."""
import importlib
import sys
from encoder.pretrained_registry import ROOT

def build(self, folder, meta, metadata, pretrained, with_decoder, method, size):
    sys.path.insert(0, str(ROOT / 'resources/third_party/sparsh'))
    module = importlib.import_module('tactile_ssl.model.vision_transformer')
    self.embed_dim = 384 if size == 'S' else 768
    # New exports use square inputs; restore the resolution saved by older policies.
    image_size = (metadata or {}).get('image_size', (224, 224))
    self.hw = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    self.model = module.VisionTransformer(img_size=self.hw, in_chans=3 if method == 'V-JEPA' else 12,
        embed_dim=self.embed_dim, depth=12, num_heads=6 if size == 'S' else 12,
        pos_embed_fn='sinusoidal', num_register_tokens=0 if method == 'MAE' else 1,
        num_frames=4 if method == 'V-JEPA' else 1, tubelet_size=2)
    if pretrained:
        from safetensors.torch import load_file
        paths = list(folder.glob('*.safetensors'))
        if len(paths) != 1:
            raise FileNotFoundError(f'Expected one official Sparsh weight file in {folder}')
        weights = load_file(paths[0])
        key = 'patch_embed.proj.weight'
        if method != 'V-JEPA':
            # Preserve original 2-frame response on repeated pairs; all 4 frames participate.
            weights[key] = weights[key].repeat(1, 2, 1, 1) / 2
        # DINO's learned corruption token belongs to pretraining, not inference.
        weights.pop('masktoken', None)
        self.model.load_state_dict(weights, strict=True)
