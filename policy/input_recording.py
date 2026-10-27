"""Record the visual tensors actually passed to unified policy inference.

Videos repeat the last inference input while its queued actions execute. Raw
diagnostic HDF5 observations use that same input's timestamp, not the later
post-action simulator observation. No Isaac Sim imports are needed here.
"""
from pathlib import Path
import json

import cv2
import numpy as np
import torch


def snapshot_observation(observation):
    def copy(value):
        if isinstance(value, dict):
            return {k: copy(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().copy()
        return np.asarray(value).copy()
    return {key: copy(observation[key]) for key in
            ('tactile', 'observation', 'embodiment', 'actor') if key in observation}


class PolicyInputRecorder:
    def __init__(self, run, save_raw=False, record_markers=False):
        self.run = run
        self.save_raw = save_raw
        self.record_markers = record_markers
        self.marker_h5 = None
        self.marker_path = None
        self.marker_count = 0
        self.keys = list(run['camera_keys'])
        options = run['options']
        self.mode = options.get('tactile_input_mode', 'marker_rgb')
        if options.get('tactile_mode', 'none') != 'none':
            self.keys.extend(f'observation.images.tac_{s}' for s in ('left','right'))
            if options.get('tactile_mode') == 'native':
                self.keys.extend(f'observation.images.tac_{s}.baseline' for s in ('left','right'))
        self.size = 256 if run['policy'] == 'act' else options.get('image_size',224)
        if run['policy'] == 'n0vtla':
            self.size = 224
        self.video_size = (self.size*len(self.keys),self.size+32)
        self.h5 = None
        self.path = None
        self.frame_count = 0

    def frame(self, images, action_index, source_action):
        canvas = np.zeros((self.video_size[1],self.video_size[0],3),np.uint8)
        for i,key in enumerate(self.keys):
            value = images[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            history_length = None
            if value.ndim == 5:
                # Temporal inputs are chronological [B,T,C,H,W]. Display the
                # newest frame; raw HDF5 recording retains the full history.
                history_length = value.shape[1]
                value = value[:, -1]
            if value.ndim != 4:
                raise ValueError(f'Unexpected actual policy image {key}: {value.shape}')
            image = value[0].transpose(1,2,0)
            if image.shape != (self.size,self.size,3):
                raise ValueError(f'Unexpected actual policy image {key}: {image.shape}')
            if not np.isfinite(image).all() or image.min() < -1e-5 or image.max() > 1.00001:
                raise ValueError(f'Policy visual input {key} is not finite [0,1]')
            canvas[32:,i*self.size:(i+1)*self.size] = np.rint(np.clip(image,0,1)*255).astype(np.uint8)
            label = key.removeprefix('observation.images.')
            if 'tac_' in label:
                label += ' '+self.mode
            if history_length is not None:
                label += f' latest/{history_length}'
            cv2.putText(canvas,label,(i*self.size+4,12),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
            cv2.putText(canvas,f'action {action_index}; input {source_action}',
                        (i*self.size+4,26),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
        return torch.from_numpy(canvas)

    def write(self, task, images, raw_observation, action_index, source_action):
        period = task.cfg.video_frequency
        if period > 0 and action_index % period == 0:
            task.video_handler.write(self.frame(images,action_index,source_action))
        if not self.save_raw:
            return
        if self.h5 is None:
            import h5py
            self.path = Path(task.save_root)/'policy_inputs'/f'{task.cfg.seed}.hdf5'
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.h5 = h5py.File(self.path,'w')
            self.h5.attrs['checkpoint'] = str(self.run['checkpoint'])
            self.h5.attrs['tactile_input_mode'] = self.mode
            if self.mode in ('depth_deform_20', 'depth_deform_80', 'depth_deform_160', 'depth_deform_240', 'depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
                from encoder.linear_depth_deform import depth_deform_spec
                self.h5.attrs['depth_deform_encoding'] = json.dumps(depth_deform_spec(self.mode))
            self.h5.attrs['image_keys'] = json.dumps(self.keys)
            self.h5.attrs['time_contract'] = 'All arrays correspond to source_action_index; queued actions repeat their inference inputs.'
            self.frame_count = 0

        def append(name,value):
            if isinstance(value,dict):
                for key,item in value.items():
                    append(f'{name}/{key}' if name else key,item)
                return
            if isinstance(value,torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if name not in self.h5:
                self.h5.create_dataset(name,shape=(0,*value.shape),maxshape=(None,*value.shape),
                                       dtype=value.dtype,chunks=(1,*value.shape),compression='lzf')
            ds = self.h5[name]
            ds.resize(self.frame_count+1,axis=0)
            ds[-1] = value
        append('action_index',np.int64(action_index))
        append('source_action_index',np.int64(source_action))
        append('policy_input',{key:value[0] for key,value in images.items()})
        if raw_observation is None:
            raise ValueError('Raw input recording requires an observation snapshot at inference time')
        append('',raw_observation)
        self.frame_count += 1
        self.h5.flush()

    def write_markers(self, task, observation, action_index, source_action):
        """Save live pre-action markers every action, including queued actions.

        marker_full is [reference/current, stable ID order, x/y] in sensor pixels.
        Out-of-view coordinates are retained without clipping or replacement.
        """
        if not self.record_markers:
            return
        payload = {'action_index': np.int64(action_index),
                   'source_action_index': np.int64(source_action)}
        for side in ('left', 'right'):
            stream = observation['tactile'][f'{side}_tactile']
            for key in ('marker_full', 'marker_id', 'marker_in_view'):
                if key not in stream:
                    raise ValueError(f'Marker tracing requires {side}_tactile/{key}')
                value = stream[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                payload[f'tactile/{side}_tactile/{key}'] = np.asarray(value)
        if self.marker_h5 is None:
            import h5py
            from encoder.linear_depth_deform import depth_deform_spec
            self.marker_path = Path(task.save_root)/'marker_traces'/f'{task.cfg.seed}.hdf5'
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            self.marker_h5 = h5py.File(self.marker_path, 'w')
            self.marker_h5.attrs['checkpoint'] = str(self.run['checkpoint'])
            self.marker_h5.attrs['tactile_input_mode'] = self.mode
            if self.mode in ('depth_deform_20', 'depth_deform_80', 'depth_deform_160', 'depth_deform_240', 'depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
                self.marker_h5.attrs['depth_deform_encoding'] = json.dumps(depth_deform_spec(self.mode))
            self.marker_h5.attrs['time_contract'] = 'Live pre-action markers at action_index; policy input last refreshed at source_action_index.'
            self.marker_h5.attrs['marker_layout'] = '[reference/current, stable marker ID order, x/y]; pixels; no clipping'
            self.marker_count = 0
        for name, value in payload.items():
            if name not in self.marker_h5:
                self.marker_h5.create_dataset(name, shape=(0, *value.shape),
                    maxshape=(None, *value.shape), dtype=value.dtype,
                    chunks=(1, *value.shape), compression='lzf')
            ds = self.marker_h5[name]
            ds.resize(self.marker_count+1, axis=0)
            ds[-1] = value
        self.marker_count += 1
        self.marker_h5.flush()

    def close(self,result=None):
        for handle_name, path_name in (('h5', 'path'), ('marker_h5', 'marker_path')):
            handle = getattr(self, handle_name)
            if handle is not None:
                if result is not None:
                    handle.attrs['result'] = result
                handle.close()
                setattr(self, handle_name, None)
                if result is not None:
                    path = getattr(self, path_name)
                    path.rename(path.with_name(path.stem+'_'+result+'.hdf5'))
