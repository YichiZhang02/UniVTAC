"""Bound persistent-worker caches while preserving temporal RGB baselines."""
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from encoder.image_io import encode_rgb, ENCODING_ATTR, STANDARD_ENCODING
from policy.train_unified import UniVTACDataset


class DatasetCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.files = []
        for index in range(4):
            path = Path(self.temp.name) / f'{index}.hdf5'
            with h5py.File(path, 'w') as handle:
                handle['embodiment/joint'] = np.zeros((3, 8), np.float32)
                for side in ('left', 'right'):
                    ds = handle.create_dataset(f'tactile/{side}_tactile/rgb_marker', (3,),
                                               dtype=h5py.vlen_dtype(np.dtype('uint8')))
                    ds.attrs[ENCODING_ATTR] = STANDARD_ENCODING
                    for step in range(3):
                        ds[step] = np.frombuffer(encode_rgb(np.full((16, 16, 3), 20+index*30+step*10, np.uint8)), np.uint8)
            self.files.append(path)
        self.dataset = UniVTACDataset({'task': self.files}, cameras=[], tactile_mode='encode',
            tactile_input_mode='marker_rgb', chunk_size=1,
            stats={key: dict(mean=[0]*8, std=[1]*8) for key in ('state', 'action')},
            policy='starvla_groot', image_size=16, tactile_num_frames=4,
            tactile_pretrained_mode='tactile', max_open_files=2)

    def tearDown(self):
        for handle in self.dataset._files.values():
            handle.close()
        self.temp.cleanup()

    def test_lru_eviction_closes_handles_and_releases_baselines(self):
        original = self.dataset[3]['observation.images.tac_left'].clone()  # file 1, frame 1
        evicted = self.dataset._files[self.files[1]]
        self.dataset[0]  # file 0
        self.dataset[3]  # file 1 becomes most recently used
        self.dataset[4]  # file 2 evicts file 0
        self.assertEqual(list(self.dataset._files), [self.files[1], self.files[2]])
        self.dataset[6]  # file 3 evicts file 1
        self.assertFalse(evicted.id.valid)
        self.assertFalse(any(key[0] == self.files[1] for key in self.dataset._baselines))
        self.assertLessEqual(len(self.dataset._files), 2)
        self.assertLessEqual(len(self.dataset._baselines), 4)
        torch.testing.assert_close(self.dataset[3]['observation.images.tac_left'], original, rtol=0, atol=0)

    def test_spawn_state_drops_live_hdf5_objects_and_cached_images(self):
        expected = self.dataset[1]['observation.images.tac_left']
        restored = pickle.loads(pickle.dumps(self.dataset))
        self.assertEqual(len(restored._files), 0)
        self.assertEqual(len(restored._baselines), 0)
        try:
            torch.testing.assert_close(restored[1]['observation.images.tac_left'], expected, rtol=0, atol=0)
        finally:
            for handle in restored._files.values():
                handle.close()


if __name__ == '__main__':
    unittest.main()
