"""Shared helpers for tactile ViT pretraining methods."""
import torch
from torch import nn
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomResizedCrop

def patchify(images, patch_size=16):
    b, c, h, w = images.shape
    return images.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size) \
        .permute(0, 2, 4, 3, 5, 1).reshape(b, -1, patch_size * patch_size * c)


def decoder(dim, depth=2, heads=8):
    layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                       activation='gelu', norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
def augment(images, local=False):
    output = []
    for image in images:
        scale = (.55, .8) if local else (.8, 1.)
        i, j, h, w = RandomResizedCrop.get_params(image, scale, (0.85, 1.15))
        crop = TF.resized_crop(image, i, j, h, w, [224, 224])
        crop = TF.adjust_brightness(crop, float(torch.empty(()).uniform_(.9, 1.1)))
        output.append(crop.clamp(0, 1))
    return torch.stack(output)


def projection_head(dim, out_dim=1024):
    return nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, out_dim))


class PixelDecoder(nn.Module):
    """Reconstruct every input patch from ViT patch tokens."""
    def __init__(self, embed_dim, channels):
        super().__init__()
        self.project = nn.Linear(embed_dim, 256)
        self.decode = decoder(256)
        self.pixels = nn.Linear(256, 16 * 16 * channels)

    def forward(self, tokens):
        return self.pixels(self.decode(self.project(tokens)))
