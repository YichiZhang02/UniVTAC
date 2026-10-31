"""Conv3d grouping, joint ViT attention, future targets and checkpoint contracts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'encoder')]

import torch
from torch import nn
from encoder.factory import local_encoder, require_current_local_checkpoint
from encoder.catalog import MULTIFRAME_METHODS
from encoder.pretrained_registry import constructor_metadata
from encoder.temporal import TemporalGrid, TemporalViT


class Conv3dTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_all_four_frame_models_default_to_4in2(self):
        from encoder.vit_methods import METHODS
        for method in MULTIFRAME_METHODS:
            for size in ('S', 'B'):
                with self.subTest(method=method, size=size), torch.device('meta'):
                    # The two-block mask precomputation uses data-dependent
                    # nonzero(), unavailable on meta; one block checks the same defaults.
                    options = {'num_blocks': 1} if method == 'I-JEPA4Frame' else {}
                    model = METHODS[method](size, **options)
                    encoder = local_encoder(method, size, num_frames=4)
                    self.assertEqual(model.encoder.output_frames, 2)
                    self.assertEqual(encoder.output_frames, 2)
                    self.assertEqual(encoder.temporal_compression, '4in2')

    def test_training_entry_defaults_and_explicit_compression(self):
        from encoder import train_common
        for method, extra, expected in (
                ('MAE4Frame', [], '4in2'), ('ResNet4Frame', [], '4in2'),
                ('SD-VAE4Frame', [], '4in2'), ('MAE', [], '4in1'),
                ('MAE4Frame', ['--temporal-compression', '4in1'], '4in1')):
            with self.subTest(method=method, extra=extra), patch.object(sys, 'argv', ['train.py', *extra]), \
                    patch.object(train_common, 'run') as run:
                train_common.main(method)
                self.assertEqual(run.call_args.args[0].temporal_compression, expected)

    def test_input_tubelets_and_joint_attention(self):
        for mode, k in (('4in1', 1), ('4in2', 2)):
            with self.subTest(mode=mode):
                encoder = TemporalViT('S', image_size=32, temporal_compression=mode).eval()
                self.assertIsInstance(encoder.vit.patch_embed, nn.Conv3d)
                self.assertEqual(encoder.vit.patch_embed.kernel_size, (4//k, 16, 16))
                self.assertFalse(any('temporal_blocks' in name or 'time_compression' in name
                                     for name, _ in encoder.named_modules()))
                images = torch.rand(1, 4, 3, 32, 32, requires_grad=True)
                block_inputs = []
                hook = encoder.vit.blocks[0].register_forward_pre_hook(
                    lambda module, args: block_inputs.append(args[0].shape))
                slices = encoder.tokens_slices(images)
                hook.remove()
                self.assertEqual(slices.shape, (1, k, 5, 384))
                # Both compressed slices enter every spatial Transformer together.
                self.assertEqual(block_inputs, [torch.Size((1, k*5, 384))])
                slices.square().mean().backward()
                self.assertTrue(torch.isfinite(encoder.vit.patch_embed.weight.grad).all())
                self.assertTrue((images.grad.abs().flatten(2).sum(-1) > 0).all())
                if k == 2:
                    with torch.no_grad():
                        conv = encoder.vit.patch_embed
                        original = conv(images.transpose(1, 2))
                        changed = images.detach().clone()
                        changed[:, :2] += 1
                        updated = conv(changed.transpose(1, 2))
                        torch.testing.assert_close(original[:, :, 1], updated[:, :, 1])
                        self.assertGreater((original[:, :, 0]-updated[:, :, 0]).abs().max().item(), 0)
                        original_tokens = encoder.patch_tokens(images)
                        updated_tokens = encoder.patch_tokens(changed)
                        # Joint attention subsequently communicates across tubelets.
                        self.assertGreater((original_tokens[:, 4:]-updated_tokens[:, 4:]).abs().max().item(), 1e-6)

    def test_single_future_frame_uses_shared_kernel_and_one_slice(self):
        for mode in ('4in1', '4in2'):
            encoder = TemporalViT('S', image_size=32, temporal_compression=mode).eval()
            future = torch.rand(2, 3, 32, 32)
            conv_inputs = []
            hook = encoder.vit.patch_embed.register_forward_pre_hook(
                lambda module, args: conv_inputs.append(args[0].detach()))
            with torch.no_grad():
                slices = encoder.tokens_slices(future, single_frame_target=True)
            hook.remove()
            self.assertEqual(slices.shape, (2, 1, 5, 384))
            expected = future[:, :, None].expand(-1, -1, encoder.tubelet_size, -1, -1)
            torch.testing.assert_close(conv_inputs[0], expected)

    def test_cnn_temporal_conv_preserves_space_and_pair_order(self):
        for mode, k in (('4in1', 1), ('4in2', 2)):
            module = TemporalGrid(8, mode)
            tau = 4//k
            self.assertEqual(module.compression.kernel_size, (tau, 1, 1))
            with torch.no_grad():
                module.compression.weight.zero_()
                module.compression.bias.zero_()
                for channel in range(8):
                    for time in range(tau):
                        module.compression.weight[channel, channel, time, 0, 0] = 10**time
            grids = torch.rand(2, 4, 8, 3, 5, requires_grad=True)
            output = module(grids)
            expected = torch.stack([sum(grids[:, start+i]*10**i for i in range(tau))
                                    for start in range(0, 4, tau)], dim=1)
            torch.testing.assert_close(output, expected)
            self.assertEqual(output.shape, (2, k, 8, 3, 5))
            output.sum().backward()
            self.assertTrue((grids.grad > 0).all())

    def test_obsolete_checkpoints_are_rejected_without_fallback(self):
        for method in ('RAE4Frame', 'RVAE4Frame', 'ResNet4Frame', 'SD-VAE4Frame'):
            state = dict(method=method, sd_patch_size=2)
            with self.assertRaisesRegex(ValueError, 'temporal_backend=conv3d'):
                require_current_local_checkpoint(state)
            state['temporal_backend'] = 'attention'
            with self.assertRaisesRegex(ValueError, 'temporal_backend=conv3d'):
                require_current_local_checkpoint(state)
            state['temporal_backend'] = 'conv3d'
            require_current_local_checkpoint(state)
            self.assertEqual(constructor_metadata(state)['temporal_backend'], 'conv3d')
        with self.assertRaisesRegex(ValueError, 'sd_patch_size=2'):
            require_current_local_checkpoint(dict(method='SD-VAE'))


if __name__ == '__main__':
    unittest.main()
