"""Verify recording uses actual policy images and preserves queued-input timing."""
from collections import deque
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

from policy.input_recording import PolicyInputRecorder
from policy.unified_deploy import Policy


def make_policy(mode):
    p = Policy.__new__(Policy)
    p.kind = 'starvla_groot'
    p.image_size = 224
    p.device = torch.device('cpu')
    p.task_name = 'pull_out_key'
    p.camera_keys = ['observation.images.cam_high','observation.images.cam_wrist']
    p.tactile_mode = 'none' if mode == 'none' else 'encode'
    p.tactile_input_mode = mode
    p.stats = {key:dict(mean=[0]*8,std=[1]*8) for key in ('state','action')}
    p.action_mode = 'absolute_joint'
    p.tokenizer = None
    p.record_policy_video = True
    p.save_policy_inputs = True
    p.temporal_agg = False
    p.chunk_size = p.execution_horizon = 2
    p.model = SimpleNamespace(predict_action_chunk=lambda batch:torch.zeros(1,2,8))
    p.reset()
    run = dict(camera_keys=p.camera_keys, policy=p.kind,checkpoint=Path('synthetic.pt'),
               options=dict(tactile_mode=p.tactile_mode,tactile_input_mode=mode,image_size=224))
    return p,run


def observation():
    image = np.zeros((240,320,3),np.uint8)
    image[...,0] = 31
    image[...,1] = 97
    image[...,2] = 181
    x,y = np.meshgrid(np.linspace(40,280,9),np.linspace(35,205,7))
    fixed = np.column_stack((x.ravel(),y.ravel())).astype(np.float32)
    stream = dict(rgb=image,rgb_marker=255-image,depth=np.full((240,320),27,np.float32),
                  marker=np.stack((fixed,fixed+[8,-5])).astype(np.float32))
    return dict(embodiment=dict(joint=np.zeros(9,np.float32)),
                observation=dict(head=dict(rgb=image),wrist=dict(rgb=image)),
                tactile={f'{side}_tactile':{k:v.copy() for k,v in stream.items()} for side in ('left','right')})


class RecordingTest(unittest.TestCase):
    def test_temporal_video_uses_latest_frame_without_modifying_history(self):
        _, run = make_policy('depth_deform_nonlinear_radial')
        recorder = PolicyInputRecorder(run)
        for frames in (4, 5):
            images = {key: torch.zeros(1, 3, 224, 224) for key in recorder.keys}
            history = torch.zeros(1, frames, 3, 224, 224)
            history[:, -1] = .75
            images['observation.images.tac_left'] = history
            original = history.clone()
            frame = recorder.frame(images, 0, 0).numpy()
            np.testing.assert_array_equal(frame[32:, 448:672], np.full((224, 224, 3), 191, np.uint8))
            torch.testing.assert_close(history, original)

    def test_pixels_match_actual_adapter_for_every_mode(self):
        for mode in ('none','rgb_only','marker_only','marker_rgb','depth_deform_20','depth_deform_80','depth_deform_160','depth_deform_240','depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
            with self.subTest(mode=mode):
                policy,run = make_policy(mode)
                batch = policy.encode_obs(observation())
                recorder = PolicyInputRecorder(run)
                images = {k:v for k,v in batch.items() if k.startswith('observation.images.')}
                frame = recorder.frame(images,0,0).numpy()
                self.assertEqual(len(recorder.keys),2 if mode=='none' else 4)
                for i,key in enumerate(recorder.keys):
                    expected = np.rint(images[key][0].permute(1,2,0).numpy()*255).astype(np.uint8)
                    np.testing.assert_array_equal(frame[32:,i*224:(i+1)*224],expected)

    def test_queued_actions_keep_original_input_and_raw_observation(self):
        for mode in ('depth_deform_80', 'depth_deform_nonlinear_radial'):
            with self.subTest(mode=mode):
                self.check_queued_actions(mode)

    def check_queued_actions(self, mode):
        policy,run = make_policy(mode)
        recorder = PolicyInputRecorder(run,save_raw=True)
        with tempfile.TemporaryDirectory() as directory:
            task = SimpleNamespace(policy_input_recorder=recorder,save_root=Path(directory),
                                   cfg=SimpleNamespace(video_frequency=0,seed=7),device='cpu',
                                   take_action=lambda *a,**kw:None)
            obs = observation()
            policy.eval(task,obs)
            obs['tactile']['left_tactile']['marker'][1,:,0] += 5
            policy.eval(task,obs)  # queued action: must retain input from action 0
            policy.eval(task,obs)  # replan: now sees the changed markers
            recorder.close('checked')
            with h5py.File(Path(directory)/'policy_inputs/7_checked.hdf5') as f:
                import json
                from encoder.linear_depth_deform import depth_deform_spec
                self.assertEqual(json.loads(f.attrs['depth_deform_encoding']), depth_deform_spec(mode))
                self.assertEqual(f['source_action_index'][:].tolist(),[0,0,2])
                self.assertEqual(f['action_index'][:].tolist(),[0,1,2])
                m = f['tactile/left_tactile/marker'][:]
                np.testing.assert_array_equal(m[0],m[1])
                np.testing.assert_allclose(m[2,1,:,0]-m[0,1,:,0],5)
                image = f['policy_input/observation.images.tac_left'][:]
                np.testing.assert_array_equal(image[0],image[1])
                self.assertFalse(np.array_equal(image[0],image[2]))

    def test_marker_trace_is_live_complete_and_keeps_ids_without_raw_recording(self):
        for mode in ('depth_deform_20', 'depth_deform_80', 'depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
            with self.subTest(mode=mode):
                self.check_marker_trace(mode)

    def check_marker_trace(self, mode):
        policy,run = make_policy(mode)
        policy.save_policy_inputs = False
        recorder = PolicyInputRecorder(run, record_markers=True)
        obs = observation()
        for stream in obs['tactile'].values():
            stream['marker_full'] = stream['marker'].copy()
            stream['marker_full'][1,0] = [-35, 400]
            stream['marker_id'] = np.arange(63, dtype=np.int64)
            stream['marker_in_view'] = np.ones(63, dtype=bool)
            stream['marker_in_view'][0] = False
        with tempfile.TemporaryDirectory() as directory:
            task = SimpleNamespace(policy_input_recorder=recorder,save_root=Path(directory),
                cfg=SimpleNamespace(video_frequency=0,seed=9),device='cpu',
                take_action=lambda *a,**kw:None)
            policy.eval(task,obs)
            obs['tactile']['left_tactile']['marker_full'][1,0] = [-45, 410]
            policy.eval(task,obs)
            recorder.close('checked')
            self.assertFalse((Path(directory)/'policy_inputs').exists())
            with h5py.File(Path(directory)/'marker_traces/9_checked.hdf5') as f:
                self.assertEqual(f['source_action_index'][:].tolist(),[0,0])
                self.assertEqual(f['action_index'][:].tolist(),[0,1])
                g = f['tactile/left_tactile']
                np.testing.assert_array_equal(g['marker_full'][:,1,0], [[-35,400],[-45,410]])
                np.testing.assert_array_equal(g['marker_id'][:], np.tile(np.arange(63), (2,1)))
                self.assertFalse(g['marker_in_view'][:,0].any())
                from encoder.linear_depth_deform import depth_deform_spec
                import json
                self.assertEqual(json.loads(f.attrs['depth_deform_encoding']), depth_deform_spec(mode))


if __name__=='__main__':
    unittest.main()
