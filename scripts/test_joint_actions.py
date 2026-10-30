"""Verify joint action semantics across dataset, statistics, and execution."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.joint_actions import encode_joint_actions, decode_joint_actions, joint_action_mode
from policy.train_unified import UniVTACDataset, compute_stats, relative_action_stats, make_config
from policy.unified_deploy import Policy
from policy.checkpoints import load_run, training_options


class JointActionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "episode.hdf5"
        self.joints = np.array([[1, 2, 3, 4, 5, 6, 7, .01, .01],
                                [2, 4, 6, 8, 10, 12, 14, .02, .02],
                                [4, 7, 10, 13, 16, 19, 22, .03, .03],
                                [8, 12, 16, 20, 24, 28, 32, .04, .04]], np.float32)
        with h5py.File(self.path, "w") as f:
            f["embodiment/joint"] = self.joints
        self.files = {"test": [self.path]}
        self.stats = {name: {"mean": [0.] * 8, "std": [1.] * 8,
                             "q01": [-1.] * 8, "q99": [1.] * 8} for name in ("state", "action")}

    def test_roundtrip_and_gripper(self):
        for tensor in (np.asarray, torch.tensor):
            state = tensor(self.joints[:2, :8])
            absolute = tensor(np.stack([self.joints[1:3, :8], self.joints[2:4, :8]]))
            for mode in ("absolute", "relative"):
                encoded = encode_joint_actions(state, absolute, mode)
                restored = decode_joint_actions(state, encoded, mode)
                np.testing.assert_allclose(restored, absolute)
                np.testing.assert_array_equal(encoded[..., 7], absolute[..., 7])
            np.testing.assert_array_equal(state, self.joints[:2, :8])

    def dataset(self, mode, kind="act"):
        dataset = UniVTACDataset(self.files, cameras=[], tactile_mode="none", tactile_input_mode="rgb_only",
                                chunk_size=3, stats=self.stats, policy=kind, image_size=224, action_mode=mode)
        self.addCleanup(lambda: [f.close() for f in dataset._files.values()])
        return dataset

    def test_dataset_anchor_and_padding(self):
        for kind in ("act", "pi05", "starvla_groot", "starvla_pi"):
            absolute, relative = self.dataset("absolute", kind), self.dataset("relative", kind)
            for t in range(3):
                target = absolute[t]["action"].numpy().copy()
                expected = target.copy()
                expected[:, :7] -= self.joints[t, :7]
                np.testing.assert_allclose(relative[t]["action"], expected)
                torch.testing.assert_close(relative[t]["observation.state"], absolute[t]["observation.state"])
                torch.testing.assert_close(relative[t]["is_pad"], absolute[t]["is_pad"])
            self.assertEqual(relative[2]["is_pad"].tolist(), [False, True, True])

    def test_statistics_cover_all_valid_horizons(self):
        stats = compute_stats(self.files, "relative", 3)
        expected = []
        for t in range(3):
            targets = self.joints[t+1:min(t+4, 4), :8].copy()
            targets[:, :7] -= self.joints[t, :7]
            expected.extend(targets)
        expected = np.array(expected)
        for key, value in (("mean", expected.mean(0)), ("std", np.maximum(expected.std(0), .01)),
                           ("q01", np.quantile(expected, .01, axis=0)), ("q99", np.quantile(expected, .99, axis=0))):
            np.testing.assert_allclose(stats["action"][key], value, rtol=1e-6)
        self.assertEqual(stats["state"], compute_stats(self.files)["state"])

    def test_bounded_quantile_sample_retains_exact_moments(self):
        full = relative_action_stats(self.files, 3)
        sampled = relative_action_stats(self.files, 3, sample_limit=2)
        self.assertEqual(sampled, relative_action_stats(self.files, 3, sample_limit=2))
        self.assertEqual(sampled["mean"], full["mean"])
        self.assertEqual(sampled["std"], full["std"])
        self.assertTrue(np.isfinite(sampled["q01"]).all())

    def test_vla_configs_accept_relative_joint(self):
        for kind in ("pi05", "starvla_groot", "starvla_pi"):
            args = SimpleNamespace(policy=kind, image_size=224, tactile_mode="none",
                                   tactile_input_mode="rgb_only", tactile_type="cls", encoder_ckpt=None,
                                   tactile_insert_location="decoder", action_mode="relative_joint",
                                   encoder_pretrained_mode='univtac', tactile_num_frames=1, tactile_frame_stride=1,
                                   device="cpu", chunk_size=32, n_action_steps=30)
            base = yaml.safe_load((ROOT / "policy" / kind / "base_config.yml").read_text())
            config = make_config(args, base, ["observation.images.cam_high"], [])
            config.validate_features()
            self.assertEqual(config.action_mode, "relative_joint")
            self.assertEqual(config.state_mode, "absolute_joint")
            self.assertEqual(config.n_action_steps, 30)

    def make_policy(self, mode, kind="starvla_groot", temporal=False):
        policy = Policy.__new__(Policy)
        policy.kind, policy.action_mode, policy.stats = kind, joint_action_mode(mode), self.stats
        policy.chunk_size, policy.execution_horizon, policy.temporal_agg = 3, 2, temporal
        chunk = torch.tensor([[[.1] * 7 + [.02], [.3] * 7 + [.03], [.5] * 7 + [.04]]])
        class Model:
            def predict_action_chunk(self, batch):
                return chunk.clone()
            def __call__(self, *args):
                return chunk.clone()
        policy.model = Model()
        policy.record_policy_video = policy.save_policy_inputs = False
        policy.encode_obs = lambda obs: {"observation.state": torch.zeros(1, 8), "camera": torch.zeros(1, 3, 2, 2)}
        policy.camera_keys, policy.tactile_mode, policy.device = ["camera"], "none", torch.device("cpu")
        policy.reset()
        return policy

    def test_deployment_denormalizes_then_decodes(self):
        observation = {"embodiment": {"joint": self.joints[0]}}
        for kind in ("act", "pi05", "starvla_groot", "starvla_pi"):
            for mode in ("absolute", "relative"):
                policy = self.make_policy(mode, kind)
                # Exercise both quantile and standard normalization inverses.
                policy.stats = {"action": {"mean": [2.] * 8, "std": [3.] * 8,
                                            "q01": [-1.] * 8, "q99": [5.] * 8}}
                chunk = policy.predict_chunk(observation).numpy()
                expected = np.array([.1] * 7 + [.02]) * 3 + 2
                if mode == "relative":
                    expected[:7] += self.joints[0, :7]
                np.testing.assert_allclose(chunk[0], expected, rtol=1e-6)

    def test_queue_keeps_original_anchor(self):
        policy = self.make_policy("relative")
        class Task:
            device = "cpu"
            def take_action(self, action, action_type):
                return action.clone()
        first = policy.eval(Task(), {"embodiment": {"joint": self.joints[0]}})
        second = policy.eval(Task(), {"embodiment": {"joint": self.joints[2]}})
        np.testing.assert_allclose(first[:7], self.joints[0, :7] + .1)
        np.testing.assert_allclose(second[:7], self.joints[0, :7] + .3)
        self.assertAlmostEqual(float(second[7]), .03)

    def test_temporal_aggregation_uses_absolute_targets(self):
        policy = self.make_policy("relative", "act", temporal=True)
        class Task:
            device = "cpu"
            def take_action(self, action, action_type):
                return action.clone()
        policy.eval(Task(), {"embodiment": {"joint": self.joints[0]}})
        result = policy.eval(Task(), {"embodiment": {"joint": self.joints[1]}})
        weights = np.exp(-.01 * np.arange(2))
        expected = np.average([self.joints[0, :7] + .3, self.joints[1, :7] + .1], axis=0, weights=weights)
        np.testing.assert_allclose(result[:7], expected, rtol=1e-6)

    def test_checkpoint_metadata_and_legacy_default(self):
        folder = Path(self.tmp.name)
        checkpoint = folder / "checkpoint_1.pt"
        checkpoint.write_bytes(b"metadata-only fixture")
        config = {"policy": "act", "camera_names": ["cam_high"], "tasks": ["test"]}
        def save():
            (folder / "train_config.yml").write_text(yaml.safe_dump(config))
            (folder / "dataset_stats.json").write_text(json.dumps(self.stats))
        save()
        self.assertEqual(load_run(checkpoint)["options"]["action_mode"], "absolute_joint")
        config["action_mode"] = "relative_joint"
        save()
        with self.assertRaisesRegex(ValueError, "require action_space"):
            load_run(checkpoint)
        self.stats["action_space"] = {"mode": "relative_joint"}
        save()
        self.assertEqual(load_run(checkpoint)["options"]["action_mode"], "relative_joint")
        self.stats["action_space"]["mode"] = "absolute_joint"
        save()
        with self.assertRaisesRegex(ValueError, "statistics disagree"):
            load_run(checkpoint)
        with self.assertRaisesRegex(ValueError, "routing disagree"):
            training_options({"action_mode": "relative_joint", "routing": {"action_mode": "absolute_joint"}})

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            joint_action_mode("relative_rot6d")


if __name__ == "__main__":
    unittest.main()
