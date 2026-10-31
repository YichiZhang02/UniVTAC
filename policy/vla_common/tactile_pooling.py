"""Policy-side spatial pooling; no parameters or checkpoint tensor keys."""
from functools import wraps
from math import isqrt

import torch
from torch.nn import functional as F


POOL_SIZES = {'pool': 3, 'pool3x3': 3, 'pool5x5': 5, 'pool7x7': 7}


def tactile_pool_size(mode):
    """Return the requested spatial side length; legacy pool means pool3x3."""
    return POOL_SIZES.get(mode)


def pool_spatial_tokens(tokens, grid_size, prefix_tokens=0, pool_size=3):
    """Pool each temporal slice to pool_size x pool_size, preserving any leading CLS tokens."""
    height, width = grid_size
    batch, count, channels = tokens.shape
    patches = tokens[:, prefix_tokens:]
    if patches.shape[1] % (height * width):
        raise ValueError(f'Cannot reshape {count} tokens into {grid_size} spatial grids')
    frames = patches.shape[1] // (height * width)
    grid = patches.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, (pool_size, pool_size)).flatten(2).transpose(1, 2)
    pooled = pooled.reshape(batch, frames * pool_size * pool_size, channels)
    return torch.cat((tokens[:, :prefix_tokens], pooled), dim=1) if prefix_tokens else pooled


def tactile_token_grid(backbone, method, pretrained_mode, token_count):
    """Return the spatial grid, independently of the number of temporal slices."""
    if pretrained_mode == 'univtac':
        from encoder.catalog import base_method
        method = base_method(method)
        side = (backbone.grid_size if method == 'SD-VAE' else 7 if method in ('VAE', 'RVAE') else isqrt(token_count // getattr(backbone, 'output_frames', 1)) if method == 'ResNet'
                else backbone.grid_size)
        return side, side
    if method in ('Wan2.2-VAE', 'Wan22_VAE'):
        side = 14 // backbone.wan_patch_size
        return side, side
    model = backbone.model
    config = getattr(model, 'config', None)
    patch = (16 if method == 'AnyTouch2' else
             getattr(model, 'patch_size', getattr(config, 'patch_size', 16)))
    ph, pw = (patch, patch) if isinstance(patch, int) else patch
    return backbone.hw[0] // ph, backbone.hw[1] // pw


def pool_tactile_tokens(tokens, backbone, method, pretrained_mode, *, includes_cls=False, pool_size=3):
    prefix = int(includes_cls)
    grid = tactile_token_grid(backbone, method, pretrained_mode, tokens.shape[1] - prefix)
    return pool_spatial_tokens(tokens, grid, prefix, pool_size)


def univtac_pool_tokens(backbone, method, images, pool_size=3):
    """Return the cls-mode feature followed by pooled spatial tokens in one pass."""
    from encoder.catalog import base_method
    multi_frame = method.endswith('4Frame')
    method = base_method(method)
    if method == 'ResNet' and multi_frame:
        grid = backbone.spatial_grid(images)
        b, k, d, h, w = grid.shape
        cls = backbone.backbone.fc(grid.mean((-2, -1))).mean(1)[:, None]
        patches = pool_spatial_tokens(grid.flatten(3).transpose(2, 3).flatten(1, 2),
                                     (h, w), pool_size=pool_size)
        patches = backbone.backbone.fc(patches)
    elif method == 'ResNet':
        net = backbone.backbone
        grid = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        grid = net.layer4(net.layer3(net.layer2(net.layer1(grid))))
        cls = net.fc(net.avgpool(grid).flatten(1))[:, None]
        patches = F.adaptive_avg_pool2d(grid, (pool_size, pool_size)).flatten(2).transpose(1, 2)
        # ResNet-B has 2048 spatial channels but a 512-D global feature.
        # Reuse its pretrained FC to put all tokens in the same space.
        patches = net.fc(patches)
    elif method in ('VAE', 'RVAE', 'SD-VAE'):
        cls, patches = backbone.cls_and_latent_tokens(images)
        side = backbone.grid_size if method == 'SD-VAE' else 7
        patches = pool_spatial_tokens(patches, (side, side), pool_size=pool_size)
    else:
        tokens = backbone.tokens(images)
        cls = tokens[:, :1]
        patches = pool_spatial_tokens(tokens[:, 1:], (backbone.grid_size, backbone.grid_size), pool_size=pool_size)
    return torch.cat((cls, patches), dim=1)


def pool_act_tactile_outputs(forward):
    """Pool ACT features and their positions without wrapping registered modules."""
    @wraps(forward)
    def pooled_forward(self, *args, **kwargs):
        pool_size = getattr(self, 'pooling_size', 3 if getattr(self, 'pooling3x3', False) else None)
        pooling = pool_size is not None and self.tactile_type == 'full'
        if pooling and self.pooling_pretrained_mode == 'tactile':
            tokens = self.backbone.pooled_tokens(args[0], pool_size)
            feature = tokens.transpose(1, 2).unsqueeze(2)
            return [feature], [self.position_embedding(feature).to(feature.dtype)]
        if (pooling and self.pooling_pretrained_mode == 'univtac'
                and hasattr(self.backbone, 'embed_dim')):
            tokens = univtac_pool_tokens(self.backbone, self.pooling_method, args[0], pool_size)
            feature = tokens.transpose(1, 2).unsqueeze(2)
            b, _, d = tokens.shape
            spatial = tokens[:, 1:].reshape(b, -1, pool_size, pool_size, d)
            k = spatial.shape[1]
            spatial = spatial.reshape(b * k, pool_size, pool_size, d).permute(0, 3, 1, 2)
            position = self.position_embedding(spatial).flatten(2)
            position = position.expand(b * k, -1, -1)
            position = position.reshape(b, k, position.shape[1], -1).permute(0, 2, 1, 3).flatten(2).unsqueeze(2)
            if hasattr(self, 'cls_position'):
                cls_position = self.cls_position.weight[None].transpose(1, 2).unsqueeze(2)
                cls_position = cls_position.expand(tokens.shape[0], -1, -1, -1)
            else:
                cls_position = position.new_zeros(tokens.shape[0], position.shape[1], 1, 1)
            return [feature], [torch.cat((cls_position, position), dim=-1).to(feature.dtype)]
        features, positions = forward(self, *args, **kwargs)
        if not pooling:
            return features, positions
        pooled_features, pooled_positions = [], []
        for feature, position in zip(features, positions):
            if feature.shape[-2] > 1:
                feature = F.adaptive_avg_pool2d(feature, (pool_size, pool_size))
                position = F.adaptive_avg_pool2d(position, (pool_size, pool_size))
            else:
                includes_cls = (self.pooling_pretrained_mode == 'univtac'
                                and self.pooling_method in ('MAE', 'DINOv2'))
                def pool(value):
                    tokens = value.squeeze(2).transpose(1, 2)
                    tokens = pool_tactile_tokens(tokens, self.backbone, self.pooling_method,
                                                 self.pooling_pretrained_mode, includes_cls=includes_cls, pool_size=pool_size)
                    return tokens.transpose(1, 2).unsqueeze(2)
                feature, position = pool(feature), pool(position)
            pooled_features.append(feature)
            pooled_positions.append(position)
        return pooled_features, pooled_positions
    return pooled_forward
