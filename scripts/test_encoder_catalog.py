"""Encoder selection, token contracts, reconstruction gradients and restore."""
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'encoder'))

import torch
from encoder.catalog import LOCAL_METHODS, EXTERNAL_METHODS, selection
from encoder.factory import local_encoder
from encoder.pretrained_registry import temporal_spec, latest_checkpoint, validate_identity


class EncoderCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_selection_matrix(self):
        for method in LOCAL_METHODS:
            for size in ('S', 'B'):
                self.assertEqual(selection('univtac', method, size, 'scratch'),
                                 ('univtac', method, size, 'scratch'))
            for initialization in ('official', 'adapted'):
                with self.assertRaises(ValueError):
                    selection('univtac', method, 'S', initialization)
        for method, (mode, _, _) in EXTERNAL_METHODS.items():
            for initialization in ('scratch', 'official', 'adapted'):
                self.assertEqual(selection('univtac', method, 'official', initialization),
                                 (mode, method, 'official', initialization))
                validate_identity(mode, method, 'official')
            for size in ('S', 'B'):
                with self.assertRaises(ValueError):
                    selection(mode, method, size)
                with self.assertRaises(ValueError):
                    validate_identity(mode, method, size)
            self.assertTrue((ROOT / 'encoder' / method / 'model.py').is_file())
            self.assertTrue((ROOT / 'encoder' / method / 'train.py').is_file())
        self.assertEqual(temporal_spec('vision', 'Wan22_VAE'), (5, 2))
        self.assertEqual(selection('univtac', 'VAE')[1], 'RVAE')

    def test_external_initialization_runs_the_right_export_or_training_path(self):
        from encoder import train_external

        class TinyEncoder(torch.nn.Module):
            def __init__(self, mode, method, size, *, metadata=None, pretrained=True, with_decoder=False):
                super().__init__()
                loaded.append(pretrained)
                self.model = torch.nn.Conv2d(3, 4, 1)
                self.method, self.embed_dim, self.hw = 'AnyTouch2', 4, (224, 224)
                self.num_frames = 1
                self.metadata = dict(pretrained_mode=mode, method=method, size=size,
                                     num_frames=1, frame_stride=1, in_chans=3, image_size=[224, 224])

            def patch_tokens(self, images):
                grid = torch.nn.functional.adaptive_avg_pool2d(images, (14, 14))
                return self.model(grid).flatten(2).transpose(1, 2)

        class TinyFrames(torch.utils.data.Dataset):
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 64

            def __getitem__(self, index):
                return torch.full((3, 224, 224), index / 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a', 'b'):
                (root / name / 'hdf5').mkdir(parents=True)
                (root / name / 'hdf5' / 'episode.hdf5').touch()
            for initialization in ('scratch', 'official', 'adapted'):
                loaded = []
                output = root / initialization
                argv = ['train_external', '--pretrained-mode', 'tactile', '--method', 'Anytouch2',
                        '--size', 'official', '--init', initialization, '--input-mode', 'marker_only',
                        '--device', 'cpu', '--data-root', str(root), '--output-dir', str(output),
                        '--workers', '0', '--epochs', '1', '--samples', '128', '--max-steps', '1',
                        '--max-val-steps', '1']
                with patch.object(sys, 'argv', argv), patch.object(train_external, 'ExternalEncoder', TinyEncoder), \
                        patch.object(train_external, 'ExternalFrames', TinyFrames):
                    train_external.main()
                self.assertEqual(loaded, [initialization != 'scratch'])
                state = torch.load(output / 'encoder.pth', weights_only=True)
                self.assertEqual(state['initialization'], initialization)
                self.assertEqual(state['size'], 'official')
                self.assertEqual((output / 'last.pth').exists(), initialization != 'official')
                if initialization != 'official':
                    self.assertEqual(state['step'], 1)
                    config = json.loads((output / 'config.json').read_text())
                    self.assertEqual(config['initialization'], initialization)

    def test_checkpoint_discovery_separates_adapted_and_legacy_rvae(self):
        with tempfile.TemporaryDirectory() as directory, patch('encoder.pretrained_registry.ROOT', Path(directory)):
            for initialization in ('scratch', 'official', 'adapted'):
                folder = Path(directory) / 'encoder_results/marker_only/tactile/Sparsh_mae/official' / initialization / 'run'
                folder.mkdir(parents=True)
                (folder / 'encoder.pth').touch()
                (folder / 'config.json').write_text(json.dumps(dict(initialization=initialization)))
                self.assertEqual(latest_checkpoint('marker_only', 'tactile', 'Sparsh_mae', 'official', initialization),
                                 folder / 'encoder.pth')
            old = Path(directory) / 'encoder_results/marker_only/univtac/VAE/S/run'
            old.mkdir(parents=True)
            (old / 'encoder.pth').touch()
            self.assertEqual(latest_checkpoint('marker_only', 'univtac', 'RVAE', 'S', 'scratch'), old / 'encoder.pth')

    def test_shell_launcher_dispatches_the_complete_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / 'python'
            launcher.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
if sys.argv[1] == '-c':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
Path(os.environ['CALL_LOG']).write_text(json.dumps(sys.argv[1:]))
''')
            launcher.chmod(0o755)
            log = root / 'arguments.json'
            environment = dict(os.environ, PYTHON_BIN=str(launcher), CALL_LOG=str(log),
                               PRETRAINED_MODE='univtac', ENCODER_SIZE='', VIT_SIZE='', NUM_PROCESSES='1')
            for method in (*LOCAL_METHODS, *EXTERNAL_METHODS):
                output = root / method
                result = subprocess.run(['bash', str(ROOT / 'train_encoder.sh'), '--method', method,
                    '--', '--output-dir', str(output)], cwd=ROOT, env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = json.loads(log.read_text())
                if method in LOCAL_METHODS:
                    self.assertTrue(any(arg.endswith(f'/{method}/train.py') for arg in command))
                    self.assertEqual(command[command.index('--init') + 1], 'scratch')
                else:
                    self.assertEqual(command[command.index('--method') + 1], method)
                    self.assertEqual(command[command.index('--size') + 1], 'official')
                    self.assertEqual(command[command.index('--init') + 1], 'official')
            for method, size, initialization in (('RAE', 'S', 'official'), ('Wan22_VAE', 'B', 'scratch')):
                log.unlink(missing_ok=True)
                result = subprocess.run(['bash', str(ROOT / 'train_encoder.sh'), '--method', method,
                    '--size', size, '--init', initialization], cwd=ROOT, env=environment, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(log.exists())

    def test_rae_reconstruction_and_policy_encoder_restore(self):
        from encoder.RAE.model import RAE
        model = RAE('S', in_chans=1)
        image = torch.rand(1, 1, 224, 224)
        tokens = model.encoder.patch_tokens(image)
        self.assertEqual(tokens.shape, (1, 196, 384))
        self.assertEqual([name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Conv2d)],
                         ['encoder.vit.patch_embed.proj'])
        self.assertFalse(any(isinstance(m, (torch.nn.ConvTranspose2d, torch.nn.Conv3d))
                             for m in model.modules()))
        loss, metrics = model(image)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('reconstruction', metrics)
        self.assertNotIn('kl', metrics)
        self.assertTrue(torch.isfinite(model.encoder.vit.patch_embed.proj.weight.grad).all())
        restored = local_encoder('RAE', 'S', 1)
        restored.load_state_dict(model.encoder.state_dict(), strict=True)
        model.encoder.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored.patch_tokens(image), model.encoder.patch_tokens(image))

    def test_rvae_keeps_legacy_checkpoint_and_49_latents(self):
        from encoder.RVAE.model import RVAEEncoder
        from VAE.model import VAEEncoder
        old = VAEEncoder('S', 1).eval()
        new = RVAEEncoder('S', 1).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        image = torch.rand(1, 1, 224, 224)
        with torch.no_grad():
            self.assertEqual(new.latent_tokens(image).shape, (1, 49, 384))
            torch.testing.assert_close(new.latent_tokens(image), old.latent_tokens(image))

    def test_sd_vae_sizes_reconstruction_and_784_latents(self):
        module = importlib.import_module('encoder.SD-VAE.model')
        counts = {}
        for size in ('S', 'B'):
            with torch.device('meta'):
                model = module.SDVAE(size)
            counts[size] = sum(p.numel() for p in model.parameters())
        self.assertTrue(20_000_000 < counts['S'] < 23_000_000, counts)
        self.assertTrue(80_000_000 < counts['B'] < 87_000_000, counts)
        model = module.SDVAE('S', 1)
        image = torch.rand(1, 1, 32, 32)
        loss, metrics = model(image)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('kl', metrics)
        self.assertTrue(torch.isfinite(model.encoder.quant_conv.weight.grad).all())
        restored = local_encoder('SD-VAE', 'S', 1).eval()
        restored.load_state_dict(model.encoder.state_dict(), strict=True)
        with torch.no_grad():
            self.assertEqual(restored.patch_tokens(torch.rand(1, 1, 224, 224)).shape, (1, 784, 4))

    def test_new_local_policy_token_modes(self):
        from policy.vla_common.tactile_encode import TactileEncoder
        for method, count in (('RAE', 196), ('RVAE', 49), ('SD-VAE', 784)):
            config = SimpleNamespace(tactile_encoder_path=None,
                tactile_encoder_config=dict(method=method, size='S', input_mode='marker_only',
                    in_chans=1, image_size=224, pretrained_mode='univtac'),
                tactile_input_mode='marker_only', tactile_type='full', freeze_tactile_encoder=False,
                tactile_encoder_keys=lambda: ['touch'])
            encoder = TactileEncoder(config, 16).eval()
            image = torch.rand(1, 1, 224, 224)
            with torch.no_grad():
                self.assertEqual(encoder({'touch': image}).shape, (1, count, 16))
                encoder.mode = 'pool3x3'
                self.assertEqual(encoder({'touch': image}).shape, (1, 10, 16))
                encoder.mode = 'cls'
                self.assertEqual(encoder({'touch': image}).shape, (1, 1, 16))

    def test_act_new_reconstruction_encoders_at_256_pixels(self):
        from policy.ACT.detr.models.backbone import ViTTactileBackbone

        class Position(torch.nn.Module):
            def forward(self, feature):
                return feature.new_zeros(feature.shape[0], 512, *feature.shape[-2:])

        for method, side in (('RAE', 14), ('RVAE', 7), ('SD-VAE', 28)):
            state = dict(method=method, size='S', input_mode='marker_only', in_chans=1, image_size=224)
            adapter = ViTTactileBackbone(method, 'S', None, Position(), 'full', 'marker_only', state).eval()
            image = torch.rand(1, 1, 256, 256)
            with torch.no_grad():
                features, positions = adapter(image)
                self.assertEqual(features[0].shape[-2:], (side, side))
                self.assertEqual(positions[0].shape, (1, 512, side, side))
                adapter.pooling_size = 3
                adapter.pooling_method = method
                adapter.pooling_pretrained_mode = 'univtac'
                features, _ = adapter(image)
                self.assertEqual(features[0].shape[-2:], (1, 10))


if __name__ == '__main__':
    unittest.main()
