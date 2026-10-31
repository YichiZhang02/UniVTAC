"""Four-frame ResNet with feature-grid time compression and clip reconstruction."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from encoder.temporal import TemporalGrid, TimeExpansion, require_clip
from encoder.reconstruction_loss import reconstruction_mse


RESNET_NAMES = {'S': 'resnet18', 'B': 'wide_resnet50_2'}


class ResNetCNNEncoder(nn.Module):
    def __init__(self, size='S', in_chans=3, temporal_compression='4in2'):
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
        self.temporal_compression = temporal_compression
        self.output_frames = int(temporal_compression[-1])
        self.num_frames = 4
        self.frame_stride = 1
        self.simulation_step_stride = 2
        self.temporal = TemporalGrid(self.backbone.fc.in_features, temporal_compression)

    def spatial_grid(self, images):
        require_clip(images)
        b, t, c, h, w = images.shape
        net = self.backbone
        x = images.reshape(b * t, c, h, w)
        if (h, w) != (224, 224):
            x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return self.temporal(x.reshape(b, t, *x.shape[1:]))

    def patch_tokens(self, images):
        return self.spatial_grid(images).flatten(3).transpose(2, 3).flatten(1, 2)

    def feature_slices(self, images):
        grid = self.spatial_grid(images)
        b, k, d, h, w = grid.shape
        return self.backbone.fc(grid.mean((-2, -1))).reshape(b, k, -1)

    def forward(self, images):
        return self.feature_slices(images).mean(1)


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
    def __init__(self, size, in_chans=3, foreground_weight=50., temporal_compression='4in2'):
        super().__init__()
        self.encoder = ResNetCNNEncoder(size, in_chans, temporal_compression=temporal_compression)
        self.decoder = ImageDecoder(in_chans)
        self.time_expansion = TimeExpansion(512, temporal_compression)
        self.foreground_weight = foreground_weight if in_chans == 1 else 0.

    def forward(self, images):
        prediction = self.decoder(self.time_expansion(self.encoder.feature_slices(images))[:, 0])
        loss = reconstruction_mse(prediction, images.flatten(0, 1), self.foreground_weight)
        return loss, {'reconstruction': loss.detach()}


Encoder = ResNetCNNEncoder
Model = ResNet
