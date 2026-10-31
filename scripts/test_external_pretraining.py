"""Initialization provenance, policy selection and external RGB input parity."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'encoder'))
from encoder.pretrained_registry import latest_checkpoint, require_initialization, constructor_metadata
from encoder.train_external import ExternalFrames, calibrate_wan, VisionAdaptation
from encoder.image_io import encode_rgb, ENCODING_ATTR, STANDARD_ENCODING


class ExternalPretrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_teacher_ema_copies_integer_position_ids(self):
        class Encoder(torch.nn.Module):
            method = 'AnyTouch2'
            embed_dim = 4

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, 2))
                self.position_ids = torch.nn.Parameter(torch.arange(3), requires_grad=False)

        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            with self.subTest(device=device):
                objective = VisionAdaptation(Encoder()).to(device)
                with torch.no_grad():
                    objective.encoder.weight.add_(2)
                    objective.encoder.position_ids.add_(5)
                objective.update_teacher(.5)
                torch.testing.assert_close(objective.teacher.weight, torch.full_like(objective.teacher.weight, 2))
                torch.testing.assert_close(objective.teacher.position_ids, objective.encoder.position_ids)
                self.assertFalse(objective.teacher.position_ids.requires_grad)

    def test_policy_discovery_separates_initialization_before_mtime(self):
        with tempfile.TemporaryDirectory() as directory, patch('encoder.pretrained_registry.ROOT', Path(directory)):
            root = Path(directory) / 'encoder_results/rgb_only/tactile/MAE/S'
            checkpoints = {}
            for init, time in [('scratch', 100), ('official', 200)]:
                folder = root / init / 'run'
                folder.mkdir(parents=True)
                (folder / 'config.json').write_text(json.dumps({'pretrained_mode': 'tactile', 'initialization': init}))
                checkpoint = folder / 'encoder.pth'
                checkpoint.touch()
                os.utime(checkpoint, (time, time))
                checkpoints[init] = checkpoint
            for init in checkpoints:
                self.assertEqual(latest_checkpoint('rgb_only', 'tactile', 'MAE', 'S', init), checkpoints[init])
            # A newer legacy official export must never enter scratch selection.
            old = root / 'legacy'
            old.mkdir()
            (old / 'encoder.pth').touch()
            self.assertEqual(latest_checkpoint('rgb_only', 'tactile', 'MAE', 'S', 'scratch'), checkpoints['scratch'])
            self.assertEqual(latest_checkpoint('rgb_only', 'tactile', 'MAE', 'S', 'official'), old / 'encoder.pth')

    def test_explicit_checkpoint_mismatch_is_rejected(self):
        for requested, actual in [('scratch', 'official'), ('official', 'scratch')]:
            with self.assertRaisesRegex(ValueError, 'initialization mismatch'):
                require_initialization({'initialization': actual}, requested)
        self.assertEqual(require_initialization({'pretrained_mode': 'tactile'}, 'official'), 'official')
        self.assertEqual(require_initialization({'pretrained_mode': 'univtac'}, 'scratch'), 'scratch')
        self.assertEqual(constructor_metadata({'initialization': 'scratch', 'encoder': {}}), {'initialization': 'scratch'})

    def test_external_rgb_history_uses_episode_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.hdf5'
            with h5py.File(path, 'w') as h5:
                ds = h5.create_dataset('tactile/left_tactile/rgb', (9,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                ds.attrs[ENCODING_ATTR] = STANDARD_ENCODING
                for index in range(9):
                    ds[index] = np.frombuffer(encode_rgb(np.full((24, 32, 3), 30+index*10, dtype=np.uint8)), dtype=np.uint8)
            dataset = ExternalFrames([path], input_mode='rgb_only', num_frames=4, background_offset=.5)
            torch.testing.assert_close(dataset[0], torch.full((4, 3, 224, 224), .5))
            expected = torch.tensor([.5+20/255, .5+40/255, .5+60/255, .5+80/255])
            torch.testing.assert_close(dataset[8].mean((1, 2, 3)), expected)
            for handle in dataset.handles.values():
                handle.close()

    def test_wan_calibration_uses_population_statistics(self):
        class FakeEncoder:
            def __init__(self):
                self.model = SimpleNamespace(config=SimpleNamespace(z_dim=2), register_to_config=self.update)
                self.metadata = {}
            def eval(self):
                return self
            def update(self, **values):
                self.model.config = values
            def posterior(self, images):
                return SimpleNamespace(mode=lambda: images)
        encoder = FakeEncoder()
        data = torch.tensor([[[[[1., 3.]]], [[[2., 6.]]]]])
        loader = torch.utils.data.DataLoader(data, batch_size=1)
        result = calibrate_wan(encoder, loader, 'cpu', 1)
        self.assertEqual(result['latent_values_per_channel'], 2)
        self.assertEqual(encoder.metadata['model_config']['latents_mean'], [2., 4.])
        self.assertEqual(encoder.metadata['model_config']['latents_std'], [1., 2.])

    def test_wan_scratch_loads_configuration_without_official_weights(self):
        from diffusers import AutoencoderKLWan
        from encoder.external import ExternalEncoder
        config = dict(AutoencoderKLWan(base_dim=8, decoder_base_dim=8,
            dim_mult=[1, 2, 2, 2], num_res_blocks=1, z_dim=2,
            latents_mean=[7., 8.], latents_std=[3., 4.]).config)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'vae').mkdir()
            (folder / 'vae/config.json').write_text(json.dumps(config, default=list))
            with patch('encoder.external.source_dir', return_value=folder), patch.object(
                    AutoencoderKLWan, 'from_pretrained', side_effect=AssertionError('Loaded official weights')):
                encoder = ExternalEncoder('vision', 'Wan2.2-VAE', 'B', pretrained=False, with_decoder=True)
                self.assertEqual(list(encoder.model.config.latents_mean), [0., 0.])
                self.assertEqual(list(encoder.model.config.latents_std), [1., 1.])
                self.assertTrue(hasattr(encoder, '_initial_decoder'))
                self.assertFalse(any('decoder' in key for key in encoder.state_dict()))


if __name__ == '__main__':
    unittest.main()
