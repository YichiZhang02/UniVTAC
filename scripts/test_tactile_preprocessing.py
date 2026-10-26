"""Check batched TPS against the established CPU representation."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from encoder.modality import read_tactile
from encoder.tactile_preprocessing import (
    collate_tactile_tps, materialize_tactile_tps, prepare_tactile_tps,
    prepare_encoder_batch, resolve_tactile_preprocessing,
)


class TactilePreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_reference(self, device):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / 'sample.h5', 'w') as h5:
                for side in ('left', 'right'):
                    group = h5.create_group(f'tactile/{side}_tactile')
                    group['depth'] = rng.uniform(22, 32, (5, 24, 32)).astype(np.float32)
                    fixed = np.array([[3, 3], [28, 3], [3, 20], [28, 20], [15, 12], [15, 12]], np.float32)
                    markers = np.zeros((5, 2, len(fixed), 2), np.float32)
                    for frame in range(5):
                        # Exercise both duplicate merging and variable control counts.
                        points = fixed.copy()
                        if frame % 2:
                            points[-1] += (2, 1)
                        markers[frame, 0] = points
                        markers[frame, 1] = points if frame == 0 else points + rng.normal(0, 35, points.shape)
                    group['marker'] = markers
                    if side == 'right':
                        group['marker_full'] = markers
                for mode in ('depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
                    for count in (1, 4, 5):
                        samples, references = [], []
                        for side in ('left', 'right'):
                            samples.append({'tactile': [prepare_tactile_tps(h5, side, i, 32) for i in range(count)],
                                            'task': 'lift can', 'state': torch.arange(8)})
                            references.append(torch.stack([read_tactile(h5, side, i, mode, 32) for i in range(count)]))
                        batch = collate_tactile_tps(samples)
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) and not k.endswith('source_hw') else v
                                 for k, v in batch.items()}
                        result = materialize_tactile_tps(batch, mode, frame_chunk=3)
                        expected = torch.stack(references)
                        if count == 1:
                            expected = expected[:, 0]
                        torch.testing.assert_close(result['tactile'].cpu(), expected, atol=2e-5, rtol=1e-5)
                        self.assertEqual(result['task'], ['lift can', 'lift can'])
                        self.assertFalse(any('.tps_' in k for k in result))
                        self.assertFalse(result['tactile'].requires_grad)

    def test_cpu_reference(self):
        self.check_reference('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_reference(self):
        self.check_reference('cuda')

    def test_invalid_grid_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / 'bad.h5', 'w') as h5:
                group = h5.create_group('tactile/left_tactile')
                group['depth'] = np.ones((1, 24, 32), np.float32)
                points = np.array([[1, 1], [2, 2], [3, 3]], np.float32)
                group['marker'] = np.stack([points, points])[None]
                with self.assertRaises(np.linalg.LinAlgError):
                    prepare_tactile_tps(h5, 'left', 0, 32)

    def test_backend_selection(self):
        self.assertEqual(resolve_tactile_preprocessing('auto', 'depth_deform_nonlinear_radial', 'cuda:1'), 'cuda')
        self.assertEqual(resolve_tactile_preprocessing('auto', 'depth_deform_nonlinear', 'cpu'), 'cpu')
        self.assertEqual(resolve_tactile_preprocessing('auto', 'rgb_only', 'cuda'), 'cpu')
        self.assertEqual(resolve_tactile_preprocessing('cpu', 'depth_deform_nonlinear', 'cuda'), 'cpu')
        with self.assertRaises(ValueError):
            resolve_tactile_preprocessing('cuda', 'rgb_only', 'cuda')

    def encoder_contracts(self, device):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'encoder'))
        from encoder.train_common import TactileFrames
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.h5'
            rng = np.random.default_rng(11)
            fixed = np.array([[3, 3], [28, 3], [3, 20], [28, 20], [15, 12], [15, 12]], np.float32)
            with h5py.File(path, 'w') as h5:
                group = h5.create_group('tactile/left_tactile')
                group['depth'] = rng.uniform(24, 31, (9, 24, 32)).astype(np.float32)
                markers = np.stack([np.stack([fixed, fixed+rng.normal(0, 4, fixed.shape)]) for _ in range(9)])
                group['marker'] = markers.astype(np.float32)
                h5['embodiment/joint'] = np.arange(9*8, dtype=np.float32).reshape(9, 8)
            for mode in ('depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
                for options in ({'num_frames': 1}, {'num_frames': 4}, {'num_frames': 5},
                                {'temporal': True}, {'action_conditioned': True, 'future_horizon': 2,
                                 'action_mean': np.zeros(8), 'action_std': np.ones(8)}):
                    cpu = TactileFrames([path], input_mode=mode, **options)
                    compact = TactileFrames([path], input_mode=mode, tactile_preprocessing='cuda', **options)
                    expected = prepare_encoder_batch(torch.utils.data.default_collate([cpu[0], cpu[3]]), device)
                    actual = prepare_encoder_batch(collate_tactile_tps([compact[0], compact[3]]), device, mode)
                    if isinstance(expected, list):
                        self.assertEqual(len(actual), len(expected))
                        for left, right in zip(actual, expected):
                            torch.testing.assert_close(left, right, atol=2e-5, rtol=1e-5)
                    else:
                        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
                    if options.get('num_frames', 1) > 1:
                        with patch.object(cpu, '_read', wraps=cpu._read) as read:
                            cpu[3]
                            self.assertEqual(read.call_count, options['num_frames'])
                    for dataset in (cpu, compact):
                        for handle in dataset.handles.values():
                            handle.close()

    def test_encoder_contracts_cpu(self):
        self.encoder_contracts('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_encoder_contracts_cuda(self):
        self.encoder_contracts('cuda')


if __name__ == '__main__':
    unittest.main()
