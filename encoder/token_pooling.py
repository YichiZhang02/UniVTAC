"""Policy-side CNN token compression, independent of pretrained RAE weights."""
import torch
from torch import nn
from torch.nn import functional as F


class Learned3x3(nn.Module):
    """Adaptive average baseline plus a trainable spatial CNN residual.

    The depthwise 3x3/stride-2 convolution starts as a binomial low-pass filter.
    The final channel-mixing convolution starts at zero: the initial output is
    exactly pool3x3, while its weights receive gradients on the first update.
    The spatial kernel begins receiving gradients once the output layer moves
    away from zero. One module is shared across both sensors and latent frames.
    """
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, 3, stride=2, padding=1,
                                 groups=channels, bias=False)
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            kernel = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16
            self.spatial.weight.copy_(kernel[None,None].expand(channels,1,3,3))
            nn.init.zeros_(self.mix.weight)

    def forward(self, features):
        if features.ndim != 4 or min(features.shape[-2:]) < 3:
            raise ValueError('Expected [B,C,H,W] features with H,W >= 3')
        baseline = F.adaptive_avg_pool2d(features,3)
        residual = F.adaptive_avg_pool2d(F.gelu(self.spatial(features)),3)
        return baseline + self.mix(residual).to(baseline.dtype)
