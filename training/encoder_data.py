"""One representation implementation for contact training and frozen evaluation."""
import bisect
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import h5py
import torch
from torch.utils.data import Dataset
from threadpoolctl import threadpool_limits

from encoder.tactile import from_group, sensor_group, representation_spec
import numpy as np


def initialize_worker(_worker_id):
    cv2.setNumThreads(1)
    threadpool_limits(limits=1)


class Files(Dataset):
    def __init__(self):
        self.handles=OrderedDict()

    def file(self,path):
        path=str(path)
        if path not in self.handles:
            if len(self.handles)>=8:
                _,handle=self.handles.popitem(last=False);handle.close()
            self.handles[path]=h5py.File(path,'r')
        self.handles.move_to_end(path)
        return self.handles[path]

    def __getstate__(self):
        state=self.__dict__.copy();state['handles']=OrderedDict();return state

    def close(self):
        for h in self.handles.values():h.close()
        self.handles.clear()


class Contact(Files):
    def __init__(self,root,frames=4,stride=2,method='tps_bilinear',grid=28):
        super().__init__()
        self.frames,self.stride,self.method,self.grid=frames,stride,method,grid
        self.items=[];self.ends=[]
        self.files=sorted(Path(root).glob('*/hdf5/*.hdf5'))
        for path in self.files:
            with h5py.File(path) as f:
                for side in ('left','right'):
                    g=sensor_group(f,side)
                    key='marker_full' if 'marker_full' in g else 'marker'
                    if not (len(g['depth'])==len(g[key])==len(g['rgb_marker'])):
                        raise ValueError(f'Inconsistent sensor lengths: {path}/{side}')
                    n=len(g['depth'])
                    if n<1:raise ValueError(f'Empty sensor stream: {path}/{side}')
                    self.items.append((path,side))
                    self.ends.append((self.ends[-1] if self.ends else 0)+n)
        if not self.items:raise ValueError(f'No contact streams: {root}')

    def __len__(self):return self.ends[-1]

    def __getitem__(self,index):
        e=bisect.bisect_right(self.ends,index);t=index-(self.ends[e-1] if e else 0)
        path,side=self.items[e];g=sensor_group(self.file(path),side)
        return torch.stack([from_group(g,max(0,t-j*self.stride),method=self.method,grid=self.grid)
                            for j in reversed(range(self.frames))])


class FrozenClips(Files):
    def __init__(self,root='data/encoder_eval_v1',split='val',method='tps_bilinear',grid=28):
        super().__init__()
        if split not in ('val','test'):raise ValueError(split)
        self.root=Path(root);self.archive=self.root/'raw_clips.h5'
        self.keys=json.loads((self.root/f'{split}.json').read_text())
        self.method,self.grid=method,grid

    def __len__(self):return len(self.keys)

    def __getitem__(self,index):
        group=self.file(self.archive)[self.keys[index]]
        return torch.stack([from_group(group,t,method=self.method,grid=self.grid) for t in range(4)])


class CachedContact(Dataset):
    """Read precomputed sensor frames; assemble the original anchor histories."""
    def __init__(self,root,frames=4,stride=2,method='tps_bilinear',grid=28):
        self.root=Path(root);self.frames,self.stride=frames,stride
        manifest=json.loads((self.root/'manifest.json').read_text())
        if not manifest.get('complete') or manifest['representation']!=representation_spec(method,grid):
            raise ValueError('Incomplete or mismatched contact representation cache')
        if manifest['dtype']!='float32':raise ValueError('Expected lossless float32 contact cache')
        self.files=[Path(r['source']['path']) for r in manifest['episodes']]
        self.items=[];self.ends=[];self.maps=OrderedDict()
        for row in manifest['episodes']:
            for sensor in row['sensors']:
                self.items.append((self.root/sensor['path'],sensor['side']))
                self.ends.append((self.ends[-1] if self.ends else 0)+sensor['frames'])
        if self.ends[-1]!=manifest['sensor_frames']:raise ValueError('Cache frame count mismatch')

    def __len__(self):return self.ends[-1]

    def __getstate__(self):
        state=self.__dict__.copy();state['maps']=OrderedDict();return state

    def __getitem__(self,index):
        e=bisect.bisect_right(self.ends,index);t=index-(self.ends[e-1] if e else 0)
        path,_=self.items[e]
        if path not in self.maps:
            if len(self.maps)>=64:self.maps.popitem(last=False)
            self.maps[path]=np.load(path,mmap_mode='r')
        self.maps.move_to_end(path)
        indices=[max(0,t-j*self.stride) for j in reversed(range(self.frames))]
        # Advanced indexing copies into an owned, contiguous array, safe for
        # worker IPC and eviction of the source memory mapping.
        return torch.from_numpy(self.maps[path][indices])

    def close(self):self.maps.clear()


class CachedFrozenClips(Dataset):
    def __init__(self,root='data/encoder_eval_v1',split='val',method='tps_bilinear',grid=28):
        self.root=Path(root)
        if split not in ('val','test'):raise ValueError(split)
        metadata=json.loads((self.root/'depth_deform_radial.json').read_text())
        if not metadata.get('complete') or metadata['representation']!=representation_spec(method,grid):
            raise ValueError('Incomplete or mismatched frozen radial cache')
        self.keys=json.loads((self.root/f'{split}.json').read_text())
        lookup={key:i for i,key in enumerate(metadata['keys'])}
        self.indices=[lookup[key] for key in self.keys]
        if metadata['dtype']!='float32':raise ValueError('Expected lossless float32 holdout cache')
        self.path=self.root/metadata['path'];self.data=None

    def __len__(self):return len(self.indices)

    def __getstate__(self):
        state=self.__dict__.copy();state['data']=None;return state

    def __getitem__(self,index):
        if self.data is None:self.data=np.load(self.path,mmap_mode='r')
        return torch.from_numpy(self.data[self.indices[index]].copy())

    def close(self):self.data=None
