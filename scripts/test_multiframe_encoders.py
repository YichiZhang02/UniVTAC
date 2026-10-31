"""Contracts for TacMind-derived four-frame scratch encoders and contact sampling.

Run: python -m unittest scripts.test_multiframe_encoders -v
CUDA is used when available; model checks use two anchors to bound memory.
"""
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'encoder')]
from encoder.catalog import MULTIFRAME_METHODS, SINGLE_FRAME_METHODS, base_method
from encoder.factory import local_encoder
from encoder.temporal import TemporalViT, augment
from encoder.train_common import TactileFrames
from encoder.vit_methods import METHODS
from encoder.pretrained_registry import constructor_metadata
from policy.vla_common.tactile_pooling import univtac_pool_tokens


class MultiFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_all_objectives_backward_export_and_pooling(self):
        for mode in ('4in1', '4in2'):
            k = int(mode[-1])
            for name in MULTIFRAME_METHODS:
                with self.subTest(method=name, compression=mode):
                    print('Checking', name, mode, flush=True)
                    torch.manual_seed(3)
                    model = METHODS[name]('S', temporal_compression=mode).to(self.device).train()
                    images = torch.rand(2, 4, 3, 224, 224, device=self.device)
                    inputs = ((images, images[:, -1], torch.zeros(2, 8, device=self.device))
                              if base_method(name) == 'V-JEPA' else (images,))
                    with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == 'cuda'):
                        loss, metrics = model(*inputs)
                    self.assertTrue(torch.isfinite(loss), metrics)
                    loss.backward()
                    temporal = [(n, p) for n, p in model.encoder.named_parameters()
                                if ('temporal' in n or 'time_' in n) and p.requires_grad]
                    self.assertTrue(temporal)
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in temporal))
                    for n, p in temporal:
                        self.assertIsNotNone(p.grad, n)
                        self.assertTrue(torch.isfinite(p.grad).all(), n)
                    if hasattr(model, 'update_teacher'):
                        self.assertFalse(model.teacher.training)
                        model.update_teacher(.9)
                    encoder = model.encoder.eval()
                    with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == 'cuda'):
                        patches = (encoder.latent_tokens(images) if base_method(name) in ('RVAE', 'SD-VAE')
                                   else encoder.patch_tokens(images))
                        feature = encoder(images)
                        pooled = univtac_pool_tokens(encoder, name, images)
                    count = 49 if base_method(name) in ('RVAE', 'ResNet') else 196
                    self.assertEqual(tuple(patches.shape[:2]), (2, count*k))
                    self.assertEqual(tuple(pooled.shape[:2]), (2, 1+9*k))
                    torch.testing.assert_close(pooled[:, 0], feature, atol=.02, rtol=.02)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'encoder.pth'
                        torch.save(dict(method=name, size='S', num_frames=4, frame_stride=1, simulation_step_stride=2,
                                        temporal_compression=mode, temporal_backend='conv3d', output_frames=k,
                                        sd_patch_size=getattr(encoder, 'sd_patch_size', 1), encoder=encoder.state_dict()), path)
                        state = torch.load(path, map_location='cpu', weights_only=True)
                        self.assertEqual(constructor_metadata(state)['temporal_compression'], mode)
                        restored = local_encoder(state['method'], state['size'], num_frames=4,
                                                 temporal_compression=state['temporal_compression'],
                                                 sd_patch_size=state['sd_patch_size']).to(self.device).eval()
                        restored.load_state_dict(state['encoder'], strict=True)
                        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == 'cuda'):
                            torch.testing.assert_close(restored(images), feature)
                    print(json.dumps(dict(method=name, compression=mode, loss=float(loss),
                                          tokens=list(patches.shape), restored=True)), flush=True)
                    del model, loss, metrics, encoder, restored, state, images, inputs, patches, feature, pooled, temporal
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()

    def test_window_boundaries_and_actual_cpu_cuda_read_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.hdf5'
            with h5py.File(path, 'w') as h5:
                h5.create_dataset('step', data=np.arange(10)*2+100)
                h5.create_dataset('tactile/left_tactile/rgb', data=np.zeros((10, 2, 2, 3), dtype=np.uint8))
                h5.create_dataset('embodiment/joint', data=np.zeros((10, 8), dtype=np.float32))
            for preprocessing in ('cpu', 'cuda'):
                data = TactileFrames([path], num_frames=4, frame_stride=1, simulation_step_stride=2, complete_history=True,
                                     action_conditioned=True, future_horizon=1, input_mode='rgb_only',
                                     action_mean=np.zeros(8), action_std=np.ones(8), tactile_preprocessing=preprocessing)
                self.assertEqual([row[2] for row in data.rows], list(range(3, 9)))
                if preprocessing == 'cpu':
                    with patch.object(data, '_read', side_effect=lambda p, side, t: torch.tensor(t)):
                        clip, future, action = data[0]
                    self.assertEqual(clip.tolist(), [0, 1, 2, 3])
                    self.assertEqual(future.item(), 4)
                else:
                    with patch('encoder.train_common.prepare_tactile_tps', side_effect=lambda h, s, t, size: t):
                        packed = data[0]
                    self.assertEqual(packed['images'], [0, 1, 2, 3])
                    self.assertEqual(packed['future_images'], [4])
                for handle in data.handles.values():
                    handle.close()
            with h5py.File(path, 'r+') as h5:
                h5['step'][5] += 1
            with self.assertRaisesRegex(ValueError, 'Discontinuous'):
                TactileFrames([path], num_frames=4, complete_history=True, frame_stride=1, simulation_step_stride=2)

    def test_temporal_order_tube_mask_and_shared_augmentation(self):
        for mode in ('4in1', '4in2'):
            encoder = TemporalViT('S', image_size=32, temporal_compression=mode).eval()
            clip = torch.rand(1, 4, 3, 32, 32)
            changed = clip.clone()
            changed[:, :, :, :16, :16] += 10
            with torch.no_grad():
                original = encoder.patch_tokens(clip)
                self.assertGreater((original-encoder.patch_tokens(clip.flip(1))).abs().max().item(), 1e-5)
                history = clip.clone(); history[:, :3] = 0
                self.assertGreater((original-encoder.patch_tokens(history)).abs().max().item(), 1e-5)
                keep = torch.tensor([[1, 2, 3]])
                torch.testing.assert_close(encoder.tokens(clip, keep), encoder.tokens(changed, keep))
                mask = torch.tensor([[True, False, False, False]])
                token = torch.zeros(1, 1, 384)
                torch.testing.assert_close(encoder.tokens(clip, patch_mask=mask, mask_token=token),
                                           encoder.tokens(changed, patch_mask=mask, mask_token=token))
            with self.assertRaises(ValueError):
                encoder(clip[:, -1])
        identical = torch.rand(2, 1, 3, 224, 224).expand(-1, 4, -1, -1, -1)
        view = augment(identical)
        torch.testing.assert_close(view[:, 0], view[:, 3])

    def test_policy_multiframe_token_modes(self):
        from policy.vla_common.tactile_encode import TactileEncoder
        for name in MULTIFRAME_METHODS:
            for mode in ('4in1', '4in2'):
                k = int(mode[-1])
                config = SimpleNamespace(tactile_encoder_path=None,
                    tactile_encoder_config=dict(method=name, size='S', input_mode='rgb_only', color_order='RGB',
                        in_chans=3, image_size=224, num_frames=4, frame_stride=1, simulation_step_stride=2,
                        temporal_compression=mode, temporal_backend='conv3d', sd_patch_size=2, pretrained_mode='univtac'),
                    tactile_input_mode='rgb_only', tactile_type='full', freeze_tactile_encoder=True,
                    tactile_encoder_keys=lambda: ['touch'])
                adapter = TactileEncoder(config, 16).eval()
                clip = torch.rand(1, 4, 3, 256, 256)
                count = 49 if base_method(name) in ('ResNet', 'RVAE') else 196
                prefix = int(base_method(name) in ('MAE', 'DINOv2'))
                with torch.no_grad():
                    self.assertEqual(tuple(adapter({'touch': clip}).shape), (1, count*k+prefix, 16))
                    adapter.mode = 'pool3x3'
                    self.assertEqual(tuple(adapter({'touch': clip}).shape), (1, 1+9*k, 16))
                    adapter.mode = 'cls'
                    self.assertEqual(tuple(adapter({'touch': clip}).shape), (1, 1, 16))
                del adapter
                gc.collect()

    def test_act_multiframe_features_and_positions(self):
        from policy.ACT.detr.models.backbone import RegisteredTactileBackbone

        class Position(torch.nn.Module):
            def forward(self, feature):
                return feature.new_zeros(feature.shape[0], 512, *feature.shape[-2:])

        for name in MULTIFRAME_METHODS:
            for mode in ('4in1', '4in2'):
                k = int(mode[-1])
                state = dict(method=name, size='S', in_chans=3, num_frames=4,
                             frame_stride=1, simulation_step_stride=2, temporal_compression=mode,
                             sd_patch_size=2, temporal_backend='conv3d', pretrained_mode='univtac')
                adapter = RegisteredTactileBackbone(state, Position(), 'full').eval()
                clip = torch.rand(1, 4, 3, 256, 256)
                count = 49 if base_method(name) in ('ResNet', 'RVAE') else 196
                with torch.no_grad():
                    features, positions = adapter(clip)
                    self.assertEqual(features[0].shape[-1], count*k)
                    self.assertEqual(positions[0].shape[-1], count*k)
                    adapter.pooling_size = 3
                    adapter.pooling_method = name
                    adapter.pooling_pretrained_mode = 'univtac'
                    features, positions = adapter(clip)
                    self.assertEqual(features[0].shape[-1], 1+9*k)
                    self.assertEqual(positions[0].shape[-1], 1+9*k)
                del adapter
                gc.collect()

    def test_simulation_stride_pretraining_policy_and_online_match(self):
        import cv2
        from policy.train_unified import UniVTACDataset
        from scripts.test_policy_input_recording import make_policy, observation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.hdf5'
            with h5py.File(path, 'w') as h5:
                h5['step'] = np.arange(10)*2+100
                h5['embodiment/joint'] = np.zeros((10, 8), np.float32)
                for key in ('observation/head/rgb', 'tactile/left_tactile/rgb', 'tactile/right_tactile/rgb'):
                    ds = h5.create_dataset(key, (10,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                    for t in range(10):
                        ds[t] = cv2.imencode('.jpg', np.full((12, 16, 3), t*10, np.uint8))[1]
            pretrain = TactileFrames([path], input_mode='rgb_only', num_frames=4,
                                     frame_stride=1, simulation_step_stride=2, complete_history=True)
            stats = {key: dict(mean=[0]*8, std=[1]*8) for key in ('state', 'action')}
            policy_data = UniVTACDataset({'task': [path]}, cameras=['head'], tactile_mode='encode',
                tactile_input_mode='rgb_only', chunk_size=2, stats=stats, policy='act', image_size=224,
                tactile_num_frames=4, tactile_frame_stride=1, tactile_simulation_step_stride=2)
            self.assertEqual([row[2] for row in policy_data.rows], list(range(3, 9)))
            policy, _ = make_policy('rgb_only')
            policy.tactile_num_frames, policy.tactile_frame_stride = 4, 1
            policy.tactile_simulation_step_stride = 2
            policy._warming_tactile_history = True
            for t in range(9):
                obs = observation()
                obs['physics_step'] = 100+2*t
                for stream in obs['tactile'].values():
                    stream['rgb'][:] = t*10
                batch = policy.encode_obs(obs)
                if t >= 3:
                    policy._warming_tactile_history = False
                    online = batch['observation.images.tac_left'][0]
                    torch.testing.assert_close(online, pretrain[t-3])
                    torch.testing.assert_close(online, policy_data[t-3]['observation.images.tac_left'])
                    self.assertEqual(list(policy.tactile_history['left'].steps), list(range(100+2*(t-3),101+2*t,2)))
            # Duplicate calls at the same physical time do not create fake history.
            torch.testing.assert_close(policy.encode_obs(obs)['observation.images.tac_left'], batch['observation.images.tac_left'])
            obs['physics_step'] += 4
            with self.assertRaisesRegex(ValueError, 'requires 2'):
                policy.encode_obs(obs)
            for handle in list(pretrain.handles.values())+list(policy_data._files.values()):
                handle.close()

    def test_online_warmup_and_queued_actions_use_real_simulation_steps(self):
        from contextlib import nullcontext
        from scripts.test_policy_input_recording import make_policy, observation
        policy, _ = make_policy('rgb_only')
        policy.tactile_num_frames, policy.tactile_frame_stride = 4, 1
        policy.tactile_simulation_step_stride = 2
        calls = []
        policy.model = SimpleNamespace(predict_action_chunk=lambda batch:
            calls.append(batch['observation.images.tac_left'].clone()) or torch.zeros(1, 2, 8))

        class Task:
            cfg = SimpleNamespace(decimation=2)
            device = 'cpu'
            physical_step = 100
            warmup_steps = 0
            def _configured_decimation(self):
                return nullcontext()
            def _step(self, is_save=False):
                self.physical_step += 2
                self.warmup_steps += 1
            def _get_observations(self):
                obs = observation()
                obs['physics_step'] = self.physical_step
                for stream in obs['tactile'].values():
                    stream['rgb'][:] = (self.physical_step-100)*5
                return obs
            def take_action(self, *args, **kwargs):
                self.physical_step += 2
        task = Task()
        for _ in range(3):
            policy.eval(task, task._get_observations())
        self.assertEqual(task.warmup_steps, 3)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0][0, :, 0, 0, 0], torch.tensor([0, 10, 20, 30])/255)
        torch.testing.assert_close(calls[1][0, :, 0, 0, 0], torch.tensor([20, 30, 40, 50])/255)
        task.cfg = SimpleNamespace(decimation=4)
        with self.assertRaisesRegex(ValueError, 'decimation=2'):
            policy.eval(task, task._get_observations())
        policy.reset()
        self.assertFalse(policy.tactile_history)

    def test_incompatible_checkpoint_stride_is_rejected(self):
        from policy.vla_common.tactile_encode import TactileEncoder
        config = SimpleNamespace(tactile_encoder_path=None,
            tactile_encoder_config=dict(method='RAE4Frame', size='S', input_mode='rgb_only',
                color_order='RGB', in_chans=3, num_frames=4, frame_stride=2, simulation_step_stride=4,
                temporal_backend='conv3d'),
            tactile_input_mode='rgb_only', tactile_type='full', freeze_tactile_encoder=True,
            tactile_encoder_keys=lambda: ['touch'])
        with self.assertRaisesRegex(ValueError, 'simulation_step_stride=2'):
            TactileEncoder(config, 16)

    def test_pipeline_refuses_untimed_single_frames(self):
        from policy.vla_common.tactile_temporal_processor import TactileTemporalWindowStep
        step = TactileTemporalWindowStep(tactile_keys=['touch'], num_frames=4,
                                        frame_offset=1, require_preassembled_window=True)
        with self.assertRaisesRegex(ValueError, 'two physical simulation steps'):
            step({'touch': torch.zeros(1, 3, 224, 224)})
        clip = torch.zeros(1, 4, 3, 224, 224)
        self.assertIs(step({'touch': clip})['touch'], clip)
        self.assertTrue(step.get_config()['require_preassembled_window'])

    def test_single_frame_state_dict_unchanged(self):
        for name in SINGLE_FRAME_METHODS:
            encoder = local_encoder(name, 'S')
            self.assertFalse(any('temporal' in key or 'time_' in key for key in encoder.state_dict()))
            encoder.load_state_dict(encoder.state_dict(), strict=True)
            del encoder
            gc.collect()


if __name__ == '__main__':
    unittest.main()
