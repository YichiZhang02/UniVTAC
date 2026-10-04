"""Two convolutional ResNet encoder sizes with same-modality reconstruction."""
import torch
from torch import nn
from torchvision import models
from reconstruction_loss import reconstruction_mse


RESNET_NAMES = {'S': 'resnet18', 'B': 'wide_resnet50_2'}


class ResNetCNNEncoder(nn.Module):
    def __init__(self, size='S', in_chans=3):
        super().__init__()
        if size not in RESNET_NAMES:
            raise ValueError(f'Expected ResNet size S or B, got {size}')
        self.size = size
        self.in_chans = in_chans
        self.backbone = getattr(models, RESNET_NAMES[size])(weights=None, num_classes=512)
        if in_chans != 3:
            self.backbone.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2,
                                            padding=3, bias=False)
        self.embed_dim = 512

    def forward(self, images):
        return self.backbone(images)


class ImageDecoder(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.fc = nn.Linear(512, 128 * 7 * 7)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, in_chans, 4, 2, 1), nn.Sigmoid())

    def forward(self, features):
        return self.layers(self.fc(features).reshape(-1, 128, 7, 7))


class ResNet(nn.Module):
    def __init__(self, size, in_chans=3, foreground_weight=50.):
        super().__init__()
        self.encoder = ResNetCNNEncoder(size, in_chans)
        self.decoder = ImageDecoder(in_chans)
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.

    def forward(self, images):
        prediction = self.decoder(self.encoder(images))
        loss = reconstruction_mse(prediction, images, self.foreground_weight)
        return loss, {'reconstruction': loss.detach()}
