"""SD-VAE latent packing, reconstruction, policy modes and checkpoint restore."""
import importlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'encoder')]

import torch
from torch import nn
from encoder.latent_patchify import patchify_latents
from encoder.pretrained_registry import constructor_metadata
from policy.vla_common.tactile_encode import TactileEncoder
from policy.ACT.detr.models.backbone import ViTTactileBackbone, RegisteredTactileBackbone


class Position(nn.Module):
    def forward(self, feature):
        # ACT's sine position encoder shares one map across the batch/time slices.
        return feature.new_zeros(1, 512, *feature.shape[-2:])


class SDVAEPatchifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_lossless_time_major_packing_and_gradients(self):
        latent = torch.arange(2*4*2*6*8, dtype=torch.float32).reshape(2, 4, 2, 6, 8).requires_grad_()
        tokens = patchify_latents(latent)
        restored = torch.empty_like(latent)
        index = 0
        for time in range(2):
            for row in range(0, 6, 2):
                for col in range(0, 8, 2):
                    expected = latent[:, :, time, row:row+2, col:col+2].reshape(2, 16)
                    torch.testing.assert_close(tokens[:, index], expected)
                    restored[:, :, time, row:row+2, col:col+2] = tokens[:, index].reshape(2, 4, 2, 2)
                    index += 1
        torch.testing.assert_close(restored, latent)
        tokens.sum().backward()
        torch.testing.assert_close(latent.grad, torch.ones_like(latent))
        torch.testing.assert_close(patchify_latents(latent, 1), latent.flatten(2).transpose(1, 2))
        torch.testing.assert_close(patchify_latents(latent[:, :, 0]), tokens[:, :12])
        with self.assertRaises(ValueError):
            patchify_latents(latent[..., :5, :])

    def test_reconstruction_uses_original_latents_and_patch_tokens_backpropagate(self):
        for method, compression in (('SD-VAE', None), ('SD-VAE4Frame', '4in1'), ('SD-VAE4Frame', '4in2')):
            with self.subTest(method=method, compression=compression):
                module = importlib.import_module(f'encoder.{method}.model')
                options = {'temporal_compression': compression} if compression else {}
                model = module.SDVAE('S', **options).to(self.device).eval()
                shape = (1, 4, 3, 32, 32) if compression else (1, 3, 32, 32)
                images = torch.rand(*shape, device=self.device)
                loss, metrics = model(images)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(model.encoder.quant_conv.weight.grad).all())
                self.assertGreater(model.encoder.quant_conv.weight.grad.abs().sum().item(), 0)
                with torch.no_grad():
                    model.encoder.latent_tokens(images)
                    repeated_loss, repeated_metrics = model(images)
                    torch.testing.assert_close(repeated_loss, loss)
                    torch.testing.assert_close(repeated_metrics['kl'], metrics['kl'])
                model.zero_grad(set_to_none=True)
                full_shape = (1, 4, 3, 224, 224) if compression else (1, 3, 224, 224)
                full_images = torch.rand(*full_shape, device=self.device, requires_grad=True)
                tokens = model.encoder.latent_tokens(full_images)
                k = int(compression[-1]) if compression else 1
                self.assertEqual(tokens.shape, (1, k*196, 16))
                tokens.square().mean().backward()
                grad_per_frame = full_images.grad.reshape(4 if compression else 1, -1).abs().sum(1)
                self.assertTrue(torch.isfinite(grad_per_frame).all())
                self.assertTrue((grad_per_frame > 0).all())

    def test_policy_modes_act_and_checkpoint_metadata(self):
        for method, compression in (('SD-VAE', None), ('SD-VAE4Frame', '4in1'), ('SD-VAE4Frame', '4in2')):
            for patch_size in (2,):
                with self.subTest(method=method, compression=compression, patch_size=patch_size):
                    state = dict(method=method, size='S', in_chans=3, input_mode='rgb_only',
                                 color_order='RGB', image_size=224, pretrained_mode='univtac',
                                 num_frames=4 if compression else 1)
                    if compression:
                        state.update(temporal_compression=compression, temporal_backend='conv3d',
                                     frame_stride=1, simulation_step_stride=2)
                    if patch_size == 2:
                        state['sd_patch_size'] = 2
                    config = SimpleNamespace(tactile_encoder_path=None, tactile_encoder_config=state,
                        tactile_input_mode='rgb_only', tactile_type='full', freeze_tactile_encoder=False,
                        tactile_encoder_keys=lambda: ['touch'])
                    vla = TactileEncoder(config, 12).to(self.device).eval()
                    self.assertEqual(config.tactile_encoder_config['sd_patch_size'], patch_size)
                    self.assertEqual(vla.proj.in_features, 4*patch_size**2)
                    k = int(compression[-1]) if compression else 1
                    side = 28 // patch_size
                    image_shape = (1, 4, 3, 224, 224) if compression else (1, 3, 224, 224)
                    images = torch.rand(*image_shape, device=self.device)
                    with torch.no_grad():
                        original = vla({'touch': images})
                        self.assertEqual(original.shape, (1, k*side**2, 12))
                        vla.mode = 'pool3x3'
                        self.assertEqual(vla({'touch': images}).shape, (1, 1+9*k, 12))
                        vla.mode = 'cls'
                        self.assertEqual(vla({'touch': images}).shape, (1, 1, 12))
                    # Serialize the path-free policy config and restore its registered weights.
                    buffer = io.BytesIO()
                    torch.save(dict(config=config.tactile_encoder_config, weights=vla.state_dict()), buffer)
                    buffer.seek(0)
                    saved = torch.load(buffer, map_location=self.device, weights_only=True)
                    config.tactile_encoder_config = constructor_metadata(saved['config'])
                    restored = TactileEncoder(config, 12).to(self.device).eval()
                    restored.load_state_dict(saved['weights'], strict=True)
                    with torch.no_grad():
                        torch.testing.assert_close(restored({'touch': images}), original)
                    del restored, vla, saved
                    act = (RegisteredTactileBackbone(state, Position(), 'full') if compression else
                           ViTTactileBackbone(method, 'S', None, Position(), 'full', 'rgb_only', state))
                    act.to(self.device).eval()
                    self.assertEqual(act.num_channels, 4*patch_size**2)
                    with torch.no_grad():
                        features, positions = act(images)
                        self.assertEqual(features[0].numel(), k*side**2*4*patch_size**2)
                        self.assertEqual(positions[0].shape[-1], features[0].shape[-1])
                        act.pooling_size = 3
                        act.pooling_method = method
                        act.pooling_pretrained_mode = 'univtac'
                        features, positions = act(images)
                        self.assertEqual(features[0].shape, (1, 4*patch_size**2, 1, 1+9*k))
                        self.assertEqual(positions[0].shape[-1], 1+9*k)


if __name__ == '__main__':
    unittest.main()
