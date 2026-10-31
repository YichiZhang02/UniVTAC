"""Causal temporal input, legacy restore, and trainable encoder integration tests."""
from pathlib import Path
import importlib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'encoder'))
from encoder.pretrained_registry import history_indices, result_root, temporal_spec
from encoder.train_common import TactileFrames, FileChunkBatchSampler
from policy.train_unified import UniVTACDataset
from scripts.test_policy_input_recording import make_policy, observation
from encoder.checkpoint_io import save_checkpoint


class TemporalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_identity_and_temporal_contract(self):
        self.assertNotEqual(result_root('marker_rgb', 'univtac', 'MAE', 'S'),
                            result_root('marker_rgb', 'tactile', 'MAE', 'S'))
        self.assertEqual(history_indices(2), [0, 0, 0, 2])
        self.assertEqual(history_indices(8), [2, 4, 6, 8])
        self.assertEqual(temporal_spec('vision', 'V-JEPA2'), (4, 2))
        self.assertEqual(temporal_spec('vision', 'DINOv3'), (1, 1))
        self.assertEqual(temporal_spec('vision', 'Wan2.2-VAE'), (5, 2))
        self.assertEqual(history_indices(8, 5, 2), [0, 2, 4, 6, 8])

    def test_wan_patchify_keeps_values_time_and_gradients(self):
        from encoder.external import patchify_latents
        latent = torch.arange(48*2*14*14, dtype=torch.float32).reshape(1,48,2,14,14).requires_grad_()
        tokens = patchify_latents(latent)
        self.assertEqual(tokens.shape, (1,98,192))
        torch.testing.assert_close(tokens[0,0], latent[0,:,0,:2,:2].reshape(-1))
        torch.testing.assert_close(tokens[0,49], latent[0,:,1,:2,:2].reshape(-1))
        restored = tokens.reshape(1,2,7,7,48,2,2).permute(0,4,1,2,5,3,6).reshape_as(latent)
        torch.testing.assert_close(restored, latent)
        tokens.sum().backward()
        torch.testing.assert_close(latent.grad, torch.ones_like(latent))

    def test_multi_gpu_loss_weights_unequal_replica_batches(self):
        from encoder.train_external import reduce_replica_loss
        losses = torch.tensor([2.,6.], requires_grad=True)
        loss, metrics = reduce_replica_loss(losses,
            {'_sample_count': torch.tensor([3.,1.]), 'reconstruction': torch.tensor([1.,5.])})
        self.assertEqual(loss.item(), 3.)
        self.assertEqual(metrics['reconstruction'].item(), 2.)
        loss.backward()
        torch.testing.assert_close(losses.grad, torch.tensor([.75,.25]))

    def test_wan_recomputation_preserves_causal_outputs_and_gradients(self):
        import copy
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock
        from encoder.wan_recompute import enable_wan_recomputation
        original = WanResidualBlock(4,4).train()
        recomputed = copy.deepcopy(original)
        enable_wan_recomputation(recomputed)
        self.assertEqual(set(original.state_dict()), set(recomputed.state_dict()))
        inputs = [torch.randn(2,4,t,6,6, requires_grad=True) for t in (1,4)]
        copies = [x.detach().clone().requires_grad_() for x in inputs]

        def run(block, frames):
            untouched = torch.randn(1)
            cache = [untouched,untouched,None,None]
            outputs = [block(x,cache,[2]) for x in frames]
            self.assertIs(cache[0],untouched)
            self.assertIs(cache[1],untouched)
            self.assertEqual(len(cache),4)
            sum(y.square().mean() for y in outputs).backward()
            return outputs

        expected, actual = run(original,inputs), run(recomputed,copies)
        for left,right in zip(expected,actual):
            torch.testing.assert_close(left,right)
        for left,right in zip(inputs,copies):
            torch.testing.assert_close(left.grad,right.grad)
        for left,right in zip(original.parameters(),recomputed.parameters()):
            torch.testing.assert_close(left.grad,right.grad)

    def test_wan_five_frames_reach_encoder_without_padding(self):
        from encoder.external import ExternalEncoder
        encoder = ExternalEncoder.__new__(ExternalEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.mode, encoder.method = 'vision', 'Wan2.2-VAE'
        encoder.num_frames, encoder.hw = 5, (4,4)
        encoder.mean, encoder.std = [0.]*3, [1.]*3
        seen = []
        encoder.model = SimpleNamespace(encode=lambda x: seen.append(x) or SimpleNamespace(latent_dist='posterior'))
        frames = torch.arange(5.).reshape(1,5,1,1,1).expand(1,5,3,4,4)
        self.assertEqual(encoder.posterior(frames), 'posterior')
        torch.testing.assert_close(seen[0], frames.transpose(1,2))
        with self.assertRaises(ValueError):
            encoder.posterior(frames[:,:4])
        encoder.num_frames = 4
        encoder.posterior(frames[:,:4])
        torch.testing.assert_close(seen[-1][:,0,:,0,0], torch.tensor([[0.,0.,1.,2.,3.]]))

    def test_file_chunks_always_form_exact_training_batches(self):
        rows = [(str(file), 'left', step) for file, count in enumerate((1,3,7,9,12,27,41,81))
                for step in range(count)]
        dataset = SimpleNamespace(rows=rows)
        sampler = FileChunkBatchSampler(dataset, 64, 42)
        batches = list(sampler)
        self.assertEqual(len(batches), len(rows)//64)
        self.assertEqual(len(sampler), len(batches))
        self.assertTrue(all(len(batch) == 64 for batch in batches))
        indices = [index for batch in batches for index in batch]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(batches, list(FileChunkBatchSampler(dataset, 64, 42)))

    def test_failed_checkpoint_write_preserves_published_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'encoder.pth'
            save_checkpoint({'value': torch.tensor(1)}, path)

            def interrupted_save(payload, temporary):
                Path(temporary).write_bytes(b'incomplete archive')
                raise OSError('simulated interrupted write')

            with patch('encoder.checkpoint_io.torch.save', side_effect=interrupted_save):
                with self.assertRaises(OSError):
                    save_checkpoint({'value': torch.tensor(2)}, path)
            self.assertEqual(torch.load(path, weights_only=True)['value'].item(), 1)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_pretraining_policy_and_online_frames_match(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trajectory.hdf5'
            images = np.stack([np.full((12, 16, 3), t*10, np.uint8) for t in range(10)])
            with h5py.File(path, 'w') as f:
                f['embodiment/joint'] = np.zeros((10, 8), np.float32)
                for key in ['observation/head/rgb', 'tactile/left_tactile/rgb', 'tactile/right_tactile/rgb']:
                    ds = f.create_dataset(key, (10,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                    for t, frame in enumerate(images):
                        ds[t] = cv2.imencode('.jpg', frame)[1]
            stats = {key: dict(mean=[0]*8, std=[1]*8) for key in ('state', 'action')}
            ds = UniVTACDataset({'task': [path]}, cameras=['head'], tactile_mode='encode',
                tactile_input_mode='rgb_only', chunk_size=2, stats=stats, policy='act', image_size=224,
                tactile_num_frames=4, tactile_frame_stride=2)
            pretrain = TactileFrames([path], input_mode='rgb_only', num_frames=4, action_conditioned=True,
                action_mean=np.zeros(8), action_std=np.ones(8))
            policy, _ = make_policy('rgb_only')
            policy.tactile_num_frames, policy.tactile_frame_stride = 4, 2
            policy.encoder_pretrained_mode = 'univtac'
            for step in range(9):
                obs = observation()
                for stream in obs['tactile'].values():
                    stream['rgb'][:] = step*10
                online = policy.encode_obs(obs)['observation.images.tac_left'][0]
                torch.testing.assert_close(online, ds[step]['observation.images.tac_left'])
                torch.testing.assert_close(online, pretrain[step][0])
            # Wan uses the same causal history API with five real observations.
            ds.tactile_num_frames = policy.tactile_num_frames = 5
            policy.reset()
            pretrain_wan = TactileFrames([path], input_mode='rgb_only', num_frames=5)
            for step in range(9):
                obs = observation()
                for stream in obs['tactile'].values():
                    stream['rgb'][:] = step*10
                online = policy.encode_obs(obs)['observation.images.tac_left'][0]
                torch.testing.assert_close(online, ds[step]['observation.images.tac_left'])
                torch.testing.assert_close(online, pretrain_wan[step])
            policy.reset()
            obs = observation()
            for stream in obs['tactile'].values():
                stream['rgb'][:] = 13
            reset = policy.encode_obs(obs)['observation.images.tac_left'][0]
            torch.testing.assert_close(reset, torch.full_like(reset, 13/255))
            for file in list(pretrain.handles.values()) + list(pretrain_wan.handles.values()) + list(ds._files.values()):
                file.close()

    def test_history_advances_during_queued_actions(self):
        policy, _ = make_policy('rgb_only')
        policy.tactile_num_frames, policy.tactile_frame_stride = 4, 2
        calls = []
        policy.model = SimpleNamespace(predict_action_chunk=lambda batch:
            calls.append(batch['observation.images.tac_left'].clone()) or torch.zeros(1,2,8))
        task = SimpleNamespace(device='cpu', take_action=lambda *a, **k: None)
        for step in range(9):
            obs = observation()
            for stream in obs['tactile'].values():
                stream['rgb'][:] = step*10
            policy.eval(task, obs)
        self.assertEqual(len(calls), 5)
        torch.testing.assert_close(calls[-1][0, :, 0, 0, 0], torch.tensor([20,40,60,80])/255)

    def test_tactile_rgb_baseline_and_reset(self):
        for offset in (.5, 130/255):
            policy, _ = make_policy('rgb_only')
            policy.encoder_pretrained_mode = 'tactile'
            policy.tactile_background_offset = offset
            policy.tactile_num_frames, policy.tactile_frame_stride = 4, 2
            first = policy.encode_obs(observation())['observation.images.tac_left']
            torch.testing.assert_close(first, torch.full_like(first, offset))
            changed = observation()
            changed['tactile']['left_tactile']['rgb'][:] = 0
            policy.reset()
            reset = policy.encode_obs(changed)['observation.images.tac_left']
            torch.testing.assert_close(reset, torch.full_like(reset, offset))

    def test_vjepa_uses_history_and_legacy_constructor_restores(self):
        Encoder = importlib.import_module('encoder.V-JEPA4Frame.model').VJEPAEncoder
        model = Encoder('S').eval()
        images = torch.rand(1,4,3,224,224)
        with torch.no_grad():
            original = model(images)
            changed = images.clone()
            changed[:, 0] = 0
            self.assertFalse(torch.allclose(original, model(changed), atol=1e-6))
            self.assertEqual(model.patch_tokens(images).shape, (1,392,384))
        Encoder = importlib.import_module('encoder.V-JEPA.model').VJEPAEncoder
        old = Encoder('S', num_frames=1)
        restored = Encoder('S', num_frames=1)
        restored.load_state_dict(old.state_dict(), strict=True)
        self.assertNotIn('temporal_embed', old.state_dict())


if __name__ == '__main__':
    unittest.main()
