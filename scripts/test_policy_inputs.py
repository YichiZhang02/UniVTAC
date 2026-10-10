"""Check collection -> training and live/offline deployment input parity."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.train_unified import UniVTACDataset
from policy.unified_deploy import Policy
from scripts.test_policy import hdf5_observation

# Load the data writer without importing Isaac Sim's environment utilities.
spec = importlib.util.spec_from_file_location("collection_data", ROOT / "resources/envs/utils/data.py")
collection_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collection_data)


class PolicyInputTests(unittest.TestCase):
    def test_training_live_and_offline_parity(self):
        for kind, image_size in (("act", 256), ("starvla_groot", 224)):
            for mode in ("marker_rgb", "rgb_only"):
                with self.subTest(policy=kind, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    self.check_parity(Path(tmp) / "demo.hdf5", kind, image_size, mode)

    def test_primary_color_channel_order(self):
        for kind, image_size in (("act", 256), ("starvla_groot", 224)):
            with self.subTest(policy=kind), tempfile.TemporaryDirectory() as tmp:
                self.check_parity(Path(tmp) / "primary.hdf5", kind, image_size, "marker_rgb",
                                  colors=([255, 0, 0], [0, 255, 0], [0, 0, 255], [240, 80, 20]))

    def check_parity(self, path, kind, image_size, mode, colors=None):
        cameras = ["head", "wrist"]
        camera_keys = ["observation.images.cam_high", "observation.images.cam_wrist"]
        tactile_key = "rgb_marker" if mode == "marker_rgb" else "rgb"
        colors = colors or ([220, 80, 20], [30, 120, 240], [50, 180, 100], [160, 40, 200])
        images = [np.full((24, 32, 3), color, np.uint8) for color in colors]
        joints = np.arange(27, dtype=np.float32).reshape(3, 9) / 100
        observation = {
            "embodiment": {"joint": torch.from_numpy(joints[0])},
            "observation": {name: {"rgb": torch.from_numpy(images[i])} for i, name in enumerate(cameras)},
            "tactile": {side + "_tactile": {tactile_key: torch.from_numpy(images[i+2])}
                        for i, side in enumerate(("left", "right"))},
        }
        collected = {
            "embodiment": {"joint": joints},
            "observation": {name: {"rgb": [images[i]] * 3} for i, name in enumerate(cameras)},
            "tactile": {side + "_tactile": {tactile_key: [images[i+2]] * 3}
                        for i, side in enumerate(("left", "right"))},
        }
        with h5py.File(path, "w") as h5:
            collection_data.HDF5Handler().dict_to_hdf5(h5, collected)
        stats = {name: {"mean": [0.2] * 8, "std": [0.3] * 8} for name in ("state", "action")}
        dataset = UniVTACDataset({"lift_can": [path]}, cameras=cameras, tactile_mode="encode",
                                tactile_input_mode=mode, chunk_size=2, stats=stats,
                                policy=kind, image_size=image_size)
        try:
            policy = Policy.__new__(Policy)
            policy.kind, policy.image_size, policy.stats = kind, image_size, stats
            policy.task_name, policy.camera_keys = "lift_can", camera_keys
            policy.tactile_mode, policy.tactile_input_mode = "encode", mode
            policy.tokenizer, policy.device = None, torch.device("cpu")
            run = {"camera_keys": camera_keys,
                   "options": {"tactile_mode": "encode", "tactile_input_mode": mode}}
            train = dataset[0]
            live = policy.encode_obs(observation)
            offline = policy.encode_obs(hdf5_observation(path, 0, run))
            self.assertEqual(live["task"], [train["task"]])
            for key, expected in train.items():
                if not key.startswith("observation."):
                    continue
                torch.testing.assert_close(offline[key][0], expected, rtol=0, atol=0)
                # Online images skip JPEG compression, which can shift each
                # channel of these constant-color patches by one intensity.
                tolerance = 1.01 / 255 if ".images." in key else 0
                torch.testing.assert_close(live[key][0], expected, rtol=0, atol=tolerance)
        finally:
            for handle in dataset._files.values():
                handle.close()


if __name__ == "__main__":
    unittest.main()
