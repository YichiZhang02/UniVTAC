"""Policy-side tactile pooling, gradients, and legacy configuration contracts."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from policy.checkpoints import training_options
from policy.vla_common.tactile_encode import TactileEncoder
from policy.vla_common.tactile_pooling import (
    pool_spatial_tokens, pool_tactile_tokens, pool_act_tactile_outputs, univtac_pool_tokens,
)


class FakeEncoder(nn.Module):
    embed_dim = 4
    grid_size = 14

    def __init__(self, *args, **kwargs):
        super().__init__()

    def tokens(self, x):
        offsets = torch.arange(197, device=x.device)[None, :, None]
        tokens = x.mean((1, 2, 3))[:, None, None].expand(-1, 197, 4) + offsets
        return tokens * 2 if getattr(self, 'method', None) == 'V-JEPA' else tokens

    def patch_tokens(self, x):
        return self.tokens(x)[:, 1:]

    def latent_tokens(self, x):
        return self.patch_tokens(x)[:, :49]

    def cls_and_latent_tokens(self, x):
        return self.tokens(x)[:, :1], self.latent_tokens(x)

    def project_cls(self, cls):
        return cls * 2

    def forward(self, x):
        return self.tokens(x)[:, 0]


def adapter(method='MAE', mode='full'):
    model = TactileEncoder.__new__(TactileEncoder)
    nn.Module.__init__(model)
    model.backbone = FakeEncoder()
    model.backbone.method = method
    model.proj = nn.Linear(4, 6)
    model.keys = ['left', 'right']
    model.pretrained_mode = 'univtac'
    model.method, model.mode = method, mode
    model.image_size, model.num_frames = 224, 1
    return model


class PoolingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_values_prefix_time_and_gradients(self):
        tokens = torch.randn(2, 1 + 2 * 7 * 7, 4, requires_grad=True)
        actual = pool_spatial_tokens(tokens, (7, 7), 1)
        expected = torch.stack([
            F.adaptive_avg_pool2d(tokens[:, 1+i*49:1+(i+1)*49].transpose(1, 2)
                                 .reshape(2, 4, 7, 7), (3, 3)).flatten(2).transpose(1, 2)
            for i in range(2)], dim=1).flatten(1, 2)
        torch.testing.assert_close(actual[:, 0], tokens[:, 0])
        torch.testing.assert_close(actual[:, 1:], expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue((tokens.grad > 0).all())

    def test_vla_full_cls_frames_and_sensors(self):
        x = torch.randn(2, 3, 3, 16, 16, requires_grad=True)
        for method, per_image in [('MAE', 10), ('DINOv2', 10), ('VAE', 10),
                                  ('I-JEPA', 10), ('V-JEPA', 10)]:
            with self.subTest(method=method):
                model = adapter(method, mode='pool')
                y = model({'left': x, 'right': x + 1})
                self.assertEqual(y.shape, (2, 2 * 3 * per_image, 6))
                y.sum().backward(retain_graph=True)
                single = x[:, 0]
                torch.testing.assert_close(model._tokens(single)[:, 0], model.backbone(single))
                model.mode = 'cls'
                pooled = model({'left': x, 'right': x})
                model.mode = 'cls'
                torch.testing.assert_close(pooled, model({'left': x, 'right': x}))
                self.assertEqual(pooled.shape, (2, 6, 6))

    def test_named_pool_sizes_values_alias_and_gradients(self):
        x = torch.randn(2, 3, 3, 16, 16, requires_grad=True)
        for method in ('MAE', 'DINOv2', 'I-JEPA', 'V-JEPA', 'VAE'):
            model = adapter(method, mode='pool')
            legacy = model({'left': x, 'right': x})
            for mode, side in [('pool3x3', 3), ('pool5x5', 5), ('pool7x7', 7)]:
                with self.subTest(method=method, mode=mode):
                    model.mode = mode
                    result = model({'left': x, 'right': x})
                    self.assertEqual(result.shape, (2, 6 * (1 + side * side), 6))
                    if side == 3:
                        torch.testing.assert_close(result, legacy)
                    single = x[:, 0]
                    resized = F.interpolate(single, (224, 224), mode='bilinear', align_corners=False)
                    patches = (model.backbone.latent_tokens(resized) if method == 'VAE'
                               else model.backbone.patch_tokens(resized))
                    grid = 7 if method == 'VAE' else 14
                    expected = F.adaptive_avg_pool2d(
                        patches.transpose(1, 2).reshape(2, 4, grid, grid), (side, side))
                    actual = model._tokens(single)
                    torch.testing.assert_close(actual[:, 1:], expected.flatten(2).transpose(1, 2))
                    torch.testing.assert_close(actual[:, 0], model.backbone(resized))
                    result.square().mean().backward(retain_graph=True)
                    self.assertTrue(torch.isfinite(x.grad).all())

    def test_external_named_pool_sizes_preserve_temporal_slices(self):
        backbone = SimpleNamespace(hw=(320, 240), model=SimpleNamespace(patch_size=16))
        model = adapter('V-JEPA')
        model.pretrained_mode = 'vision'
        model.backbone = FakeEncoder()
        model.backbone.hw = backbone.hw
        model.backbone.model = backbone.model
        tokens = torch.randn(2, 600, 4, requires_grad=True)
        with patch.object(model, '_unpooled_tokens', return_value=tokens):
            for mode, side in [('pool', 3), ('pool3x3', 3), ('pool5x5', 5), ('pool7x7', 7)]:
                model.mode = mode
                actual = model._tokens(torch.empty(0))
                expected = F.adaptive_avg_pool2d(
                    tokens.reshape(4, 20, 15, 4).permute(0, 3, 1, 2), (side, side))
                expected = expected.flatten(2).transpose(1, 2).reshape(2, 2 * side * side, 4)
                torch.testing.assert_close(actual, expected)
                actual.sum().backward(retain_graph=True)
        self.assertTrue((tokens.grad > 0).all())

    def test_disabled_unchanged_and_strict_weights(self):
        model = adapter()
        x = torch.randn(2, 3, 16, 16)
        torch.testing.assert_close(model._tokens(x), model._unpooled_tokens(x))
        state = model.state_dict()
        model.mode = 'pool'
        self.assertEqual(set(state), set(model.state_dict()))
        model.load_state_dict(state, strict=True)
        self.assertEqual(model._tokens(x).shape, (2, 10, 4))

    def test_external_rectangular_and_temporal_grids(self):
        for method, geometry, count, expected in [
            ('V-JEPA', dict(hw=(320, 240), model=SimpleNamespace(patch_size=16)), 600, 18),
            ('AnyTouch2', dict(hw=(224, 224), model=SimpleNamespace()), 392, 18),
            ('Wan2.2-VAE', dict(wan_patch_size=2), 98, 18),
            ('V-JEPA2', dict(hw=(256, 256), model=SimpleNamespace(config=SimpleNamespace(patch_size=16))), 512, 18),
        ]:
            with self.subTest(method=method):
                tokens = torch.randn(2, count, 4)
                result = pool_tactile_tokens(tokens, SimpleNamespace(**geometry), method, 'vision')
                self.assertEqual(result.shape, (2, expected, 4))
        result = pool_tactile_tokens(torch.randn(2, 64, 4), None, 'ResNet', 'univtac')
        self.assertEqual(result.shape, (2, 9, 4))

    def test_act_maps_and_flat_cls_features(self):
        class ACT(nn.Module):
            tactile_type = 'full'
            pooling3x3 = True
            pooling_pretrained_mode = 'univtac'
            pooling_method = 'MAE'
            backbone = SimpleNamespace(grid_size=14)

            @pool_act_tactile_outputs
            def forward(self, feature, pos):
                return [feature], [pos]

        model = ACT()
        flat, positions = torch.randn(2, 4, 1, 197), torch.randn(2, 8, 1, 197)
        feature, pos = model(flat, positions)
        self.assertEqual(feature[0].shape, (2, 4, 1, 10))
        self.assertEqual(pos[0].shape, (2, 8, 1, 10))
        torch.testing.assert_close(feature[0][:, :, :, 0], flat[:, :, :, 0])
        spatial = torch.randn(2, 4, 7, 7)
        feature, _ = model(spatial, spatial)
        torch.testing.assert_close(feature[0], F.adaptive_avg_pool2d(spatial, (3, 3)))
        model.tactile_type = 'cls'
        single = torch.randn(2, 4, 1, 1)
        feature, _ = model(single, single)
        self.assertIs(feature[0], single)
        model.tactile_type, model.pooling3x3 = 'full', False
        feature, _ = model(flat, positions)
        self.assertIs(feature[0], flat)

    def test_vjepa_projects_full_tokens_once_before_pooling(self):
        import importlib
        sys.path.insert(0, str(ROOT / 'encoder'))
        from vit_encoder import TactileViT
        model = importlib.import_module('V-JEPA.model').VJEPAEncoder('S').eval()
        x = torch.randn(2, 3, 224, 224)
        keys = set(model.state_dict())
        with torch.no_grad():
            raw = TactileViT.tokens(model, x)
            expected = model.projector(raw.reshape(-1, 384)).reshape(2, 197, 384)
            torch.testing.assert_close(model.tokens(x), expected)
            torch.testing.assert_close(model.patch_tokens(x), expected[:, 1:])
            torch.testing.assert_close(model(x), expected[:, 0])
            pooled = univtac_pool_tokens(model, 'V-JEPA', x)
            torch.testing.assert_close(pooled[:, 0], expected[:, 0])
            torch.testing.assert_close(pooled[:, 1:], pool_spatial_tokens(expected[:, 1:], (14, 14)))
            self.assertFalse(torch.allclose(model.patch_tokens(x), raw[:, 1:]))
        model.train()
        # Sequence projection must support BatchNorm and gradients over B*N tokens.
        tokens = torch.randn(2, 196, 384, requires_grad=True)
        model.project_tokens(tokens).square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertIsNotNone(model.projector[0].weight.grad)
        # The CLS-only path must retain the original pretraining behavior.
        with torch.no_grad():
            raw_cls = TactileViT.tokens(model, x)[:, 0]
            expected_cls = model.project_cls(raw_cls)
            bn = model.projector[1]
            before = bn.num_batches_tracked.item()
            torch.testing.assert_close(model.projected_latent(x), expected_cls)
            self.assertEqual(bn.num_batches_tracked.item(), before + 1)
            self.assertTrue(torch.isfinite(model(x[:1])).all())
        model.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), keys)

    def test_real_univtac_cls_plus_nine_and_gradients(self):
        import importlib
        sys.path.insert(0, str(ROOT / 'encoder'))
        from vit_encoder import TactileViT
        from VAE.model import VAEEncoder
        from ResNet.model import ResNetCNNEncoder
        factories = [
            ('MAE', lambda: TactileViT('S')),
            ('DINOv2', lambda: TactileViT('S')),
            ('I-JEPA', lambda: importlib.import_module('I-JEPA.model').IJEPAEncoder('S')),
            ('V-JEPA', lambda: importlib.import_module('V-JEPA.model').VJEPAEncoder('S')),
            ('VAE', lambda: VAEEncoder('S')),
            ('ResNet', lambda: ResNetCNNEncoder('S')),
            ('ResNet', lambda: ResNetCNNEncoder('B')),
        ]
        for method, factory in factories:
            with self.subTest(method=method, factory=factory):
                backbone = factory().eval()
                original_keys = set(backbone.state_dict())
                x = torch.randn(1, 3, 224, 224, requires_grad=True)
                actual = univtac_pool_tokens(backbone, method, x)
                self.assertEqual(actual.shape, (1, 10, backbone.embed_dim))
                with torch.no_grad():
                    torch.testing.assert_close(actual[:, 0], backbone(x))
                    if method == 'VAE':
                        patches = backbone.latent_tokens(x)
                        expected = pool_spatial_tokens(patches, (7, 7))
                    elif method == 'ResNet':
                        net = backbone.backbone
                        grid = net.maxpool(net.relu(net.bn1(net.conv1(x))))
                        grid = net.layer4(net.layer3(net.layer2(net.layer1(grid))))
                        expected = net.fc(F.adaptive_avg_pool2d(grid, (3, 3)).flatten(2).transpose(1, 2))
                    else:
                        expected = pool_spatial_tokens(backbone.patch_tokens(x), (14, 14))
                    torch.testing.assert_close(actual[:, 1:], expected)
                actual.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertEqual(set(backbone.state_dict()), original_keys)
                del actual, backbone, x

    def test_legacy_and_new_config_restore(self):
        self.assertEqual(training_options({})['tactile_type'], 'cls')
        self.assertEqual(training_options({'routing': {'tactile_type': 'full'}})['tactile_type'], 'full')
        self.assertEqual(training_options({'routing': {'tactile_type': 'pool'}})['tactile_type'], 'pool')
        from policy.unified_deploy import make_vla_config
        from policy.train_unified import make_config
        camera = 'observation.images.cam_high'
        for kind in ('pi05', 'starvla_groot', 'starvla_pi'):
            for mode in ('cls', 'full', 'pool', 'pool3x3', 'pool5x5', 'pool7x7'):
                with self.subTest(policy=kind, mode=mode):
                    args = SimpleNamespace(policy=kind, chunk_size=3, n_action_steps=2,
                        image_size=224, tactile_mode='encode', tactile_input_mode='rgb_only',
                        tactile_type=mode, tactile_insert_location='encoder', encoder_ckpt=Path('/tmp/unused'),
                        encoder_pretrained_mode='univtac', tactile_num_frames=1,
                        tactile_frame_stride=1, action_mode='absolute_joint', device='cpu')
                    config = make_config(args, {}, [camera], ['observation.images.tac_left'])
                    self.assertEqual(config.tactile_type, mode)
                    saved = dict(base={}, input_features={camera: [3, 224, 224], 'observation.state': [8]},
                        output_features={'action': [8]}, state_mode='absolute_joint', action_gap=1,
                        chunk_size=3, n_action_steps=2, tactile_encoder_config=dict(
                            method='MAE', size='S', image_size=224, in_chans=3,
                            input_mode='rgb_only', color_order='RGB'))
                    options = dict(tactile_mode='encode', tactile_input_mode='rgb_only', tactile_type=mode,
                        tactile_insert_location='encoder', action_mode='absolute_joint', chunk_size=3)
                    run = dict(policy=kind, directory=Path('/tmp'), config=saved, options=options,
                               camera_keys=[camera])
                    restored = make_vla_config(run, 'cpu', ROOT / 'resources/pretrained_models')
                    self.assertEqual(restored.tactile_type, mode)

    def test_act_builder_modes_and_strict_legacy_load(self):
        sys.path.insert(0, str(ROOT / 'policy/ACT/detr'))
        sys.path.insert(0, str(ROOT / 'encoder'))
        from policy.ACT.detr.models.backbone import build_tactile_backbone
        # Import real class definitions before mocking the constructor used by ACT.
        import importlib
        for module in ('VAE.model', 'I-JEPA.model', 'V-JEPA.model'):
            importlib.import_module(module)
        state = dict(method='MAE', size='S', input_mode='rgb_only', color_order='RGB',
                     image_size=224, in_chans=3)
        args = SimpleNamespace(lr_tactile_backbone=0, tactile_masks=False,
            tactile_type='full', hidden_dim=512, position_embedding='sine',
            tactile_encoder_method='MAE', tactile_encoder_config=state,
            tactile_ckpt=None, tactile_encoder_size='S', tactile_input_mode='rgb_only')
        x = torch.randn(2, 3, 16, 16)
        with patch('encoder.factory.local_encoder', side_effect=FakeEncoder):
            full = build_tactile_backbone(args)
            expected = full(x)
            for mode, side in [('pool', 3), ('pool3x3', 3), ('pool5x5', 5), ('pool7x7', 7)]:
                args.tactile_type = mode
                pooled = build_tactile_backbone(args)
                pooled.load_state_dict(full.state_dict(), strict=True)
                feature, position = pooled(x)
                self.assertEqual(feature[0].shape, (2, 4, 1, 1 + side * side))
                self.assertEqual(position[0].shape, (2, 512, 1, 1 + side * side))
                torch.testing.assert_close(feature[0][..., 0], expected[0][0][..., 0])
            args.tactile_type = 'cls'
            cls = build_tactile_backbone(args)
            self.assertEqual(cls(x)[0][0].shape, (2, 4, 1, 1))
    def test_act_resnet_b_pool_global_dimension(self):
        sys.path.insert(0, str(ROOT / 'policy/ACT/detr'))
        sys.path.insert(0, str(ROOT / 'encoder'))
        from policy.ACT.detr.models.backbone import build_tactile_backbone
        state = dict(method='ResNet', size='B', architecture='resnet', input_mode='rgb_only',
                     color_order='RGB', image_size=224, in_chans=3)
        args = SimpleNamespace(lr_tactile_backbone=0, tactile_masks=False,
            tactile_type='pool', hidden_dim=512, position_embedding='sine',
            tactile_encoder_method='ResNet', tactile_encoder_config=state,
            tactile_ckpt=None, tactile_encoder_size='B', tactile_input_mode='rgb_only')
        model = build_tactile_backbone(args).eval()
        self.assertEqual(model.num_channels, 512)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features, positions = model(x)
            self.assertEqual(features[0].shape, (1, 512, 1, 10))
            self.assertEqual(positions[0].shape, (1, 512, 1, 10))
            torch.testing.assert_close(features[0][:, :, 0, 0], model.backbone(x))



if __name__ == '__main__':
    unittest.main()
