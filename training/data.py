"""Raw Isaac51 episodes, all tasks and all episodes; no evaluation split."""
import bisect
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from training.actions import encode_actions
from encoder.tactile import REPRESENTATION, from_group, sensor_group

TASKS = ('grasp_classify','insert_HDMI','insert_hole','insert_tube','lift_bottle',
         'lift_can','pull_out_key','put_bottle_in_shelf')

def decode(value):
    if isinstance(value, (bytes, np.bytes_)):
        value = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
        if value is None:
            raise ValueError('Invalid encoded image')
        # Upstream img_to_stream encodes the RGB array directly with OpenCV.
        return value
    value = np.asarray(value)
    if value.ndim == 1:
        return decode(value.tobytes())
    return value

def image_tensor(value):
    rgb = cv2.resize(decode(value), (224,224))
    return torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255

class Episodes(Dataset):
    def __init__(self, root, instructions, chunk=50, tactile=False, frames=4, stride=2,
                 tactile_input_mode=REPRESENTATION, interpolation='tps_bilinear', interpolation_grid=28):
        self.chunk,self.tactile,self.frames,self.stride = chunk,tactile,frames,stride
        self.tactile_input_mode=tactile_input_mode
        self.interpolation,self.interpolation_grid=interpolation,interpolation_grid
        if tactile_input_mode not in (REPRESENTATION,'marker_rgb'):
            raise ValueError(tactile_input_mode)
        self.files=[]; self.ends=[]; self.lang={}; self.stats=None
        for task in TASKS:
            files=sorted((Path(root)/task/'hdf5').glob('*.hdf5'))
            if not files:
                raise ValueError(f'No episodes for {task}: {root}')
            self.lang[task]=json.loads((Path(instructions)/f'{task}.json').read_text())['seen'][0]
            for file in files:
                with h5py.File(file,'r') as f:
                    n=len(f['embodiment/joint'])-1
                if n < 1: raise ValueError(f'Empty episode: {file}')
                self.files.append(file); self.ends.append((self.ends[-1] if self.ends else 0)+n)
    def __len__(self): return self.ends[-1]
    def moments(self):
        # Every valid anchor and future target participates in train statistics.
        ss=[]; aa=[]
        for file in self.files:
            with h5py.File(file,'r') as f: q=np.asarray(f['embodiment/joint'][:,:8],np.float64)
            ss.append(q[:-1])
            for offset in range(1,self.chunk+1):
                if len(q)>offset:
                    aa.append(encode_actions(q[:-offset], q[offset:, None, :])[:, 0])
        s=np.concatenate(ss); a=np.concatenate(aa)
        self.stats=dict(state_mean=s.mean(0).tolist(), state_std=np.maximum(s.std(0),1e-3).tolist(),
                        action_mean=a.mean(0).tolist(), action_std=np.maximum(a.std(0),1e-3).tolist())
        return self.stats
    def __getitem__(self,index):
        e=bisect.bisect_right(self.ends,index); t=index-(self.ends[e-1] if e else 0)
        with h5py.File(self.files[e],'r') as f:
            q=np.asarray(f['embodiment/joint'][t,:8],np.float32)
            future=np.asarray(f['embodiment/joint'][t+1:t+self.chunk+1,:8],np.float32)
            n=len(future)
            action=encode_actions(q, np.concatenate((future,np.repeat(future[-1:],self.chunk-n,axis=0)),axis=0))
            cams=torch.stack([image_tensor(f[f'observation/{c}/rgb'][t]) for c in ('head','wrist')])
            tac=[]
            if self.tactile:
                for side in ('left','right'):
                    group=sensor_group(f,side)
                    indices=[max(0,t-i*self.stride) for i in reversed(range(self.frames))]
                    if self.tactile_input_mode == REPRESENTATION:
                        frames=[from_group(group,i,method=self.interpolation,grid=self.interpolation_grid) for i in indices]
                    else:
                        frames=[image_tensor(group['rgb_marker'][i]) for i in indices]
                    tac.append(torch.stack(frames))
        s=self.stats
        if s is not None:
            q=(q-s['state_mean'])/np.asarray(s['state_std'])
            action=(action-s['action_mean'])/np.asarray(s['action_std'])
        return dict(state=torch.as_tensor(q,dtype=torch.float32),action=torch.as_tensor(action,dtype=torch.float32),
                    is_pad=torch.arange(self.chunk)>=n,image=cams,
                    tactile=torch.stack(tac) if tac else torch.empty(0),
                    lang=self.lang[self.files[e].parents[1].name])
