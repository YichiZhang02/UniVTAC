"""Lossless cached episode and frozen split parity, including worker loading."""
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from encoder.tactile import representation_spec
from training.cache_tactile import convert_episode
from training.encoder_data import Contact,CachedContact,FrozenClips,CachedFrozenClips,initialize_worker


def main():
    torch.set_num_threads(2)
    with threadpool_limits(limits=1),tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'raw/object/hdf5/0.hdf5';source.parent.mkdir(parents=True)
        xx,yy=np.meshgrid(np.linspace(1,15,4),np.linspace(1,15,4));fixed=np.stack((xx.ravel(),yy.ravel()),-1).astype('float32')
        with h5py.File(source,'w') as f:
            for side in ('left','right'):
                g=f.create_group(f'tactile/{side}_tactile')
                g['depth']=np.stack([np.full((16,16),30-t/10,np.float32) for t in range(9)])
                g['marker']=np.stack([np.stack((fixed,fixed+t*.1)) for t in range(9)])
                g['rgb_marker']=np.array([b'placeholder']*9)
        cache=root/'cache'
        metadata=convert_episode((str(source),'object/hdf5/0.hdf5',str(cache),'tps_bilinear',28))
        assert convert_episode((str(source),'object/hdf5/0.hdf5',str(cache),'tps_bilinear',28))==metadata
        manifest=dict(complete=True,dtype='float32',representation=representation_spec(),episodes=[metadata],sensor_frames=18)
        (cache/'manifest.json').write_text(json.dumps(manifest))
        raw=Contact(root/'raw');stored=CachedContact(cache)
        assert len(raw)==len(stored)==18
        for i in range(18):torch.testing.assert_close(stored[i],raw[i],rtol=0,atol=0)
        batch=next(iter(DataLoader(stored,batch_size=4,num_workers=2,worker_init_fn=initialize_worker)))
        torch.testing.assert_close(batch,torch.stack([raw[i] for i in range(4)]),rtol=0,atol=0)
        raw.close();stored.close()
        print('PASS: sequential NPY writer, resume, exact raw/cache parity, history padding, two sensors and multiworker loading',flush=True)
    with threadpool_limits(limits=1):
        for split in ('val','test'):
            raw=FrozenClips(split=split);stored=CachedFrozenClips(split=split)
            assert raw.keys==stored.keys and len(raw)==len(stored)==80
            for i in range(len(raw)):torch.testing.assert_close(stored[i],raw[i],rtol=0,atol=0)
            raw.close();stored.close()
            print(f'PASS: all {split} clips are bitwise identical after caching',flush=True)


if __name__=='__main__':main()
