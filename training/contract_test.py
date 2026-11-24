"""Data-contract tests with a newly created synthetic eight-task dataset."""
import json
import tempfile
from pathlib import Path
import h5py
import numpy as np
import torch
from training.data import Episodes,TASKS
from training.actions import encode_actions,decode_actions

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for task in TASKS:
            folder=root/task/'hdf5';folder.mkdir(parents=True)
            with h5py.File(folder/'0.hdf5','w') as f:
                f['embodiment/joint']=np.arange(6*8,dtype=np.float32).reshape(6,8)
                for cam in ('head','wrist'): f[f'observation/{cam}/rgb']=np.zeros((6,24,24,3),np.uint8)
            (root/f'{task}.json').write_text(json.dumps({'seen':[task]}))
        ds=Episodes(root,root,chunk=50)
        assert len(ds)==8*5 and len(ds.files)==8
        first=ds[0];last=ds[4]
        assert first['action'].shape==(50,8)
        torch.testing.assert_close(first['action'][0],torch.tensor([8.]*7+[15.]))
        torch.testing.assert_close(first['action'][4],torch.tensor([40.]*7+[47.]))
        assert first['is_pad'].sum()==45 and last['is_pad'].sum()==49
        torch.testing.assert_close(last['action'],torch.tensor([8.]*7+[47.]).expand(50,8))
        anchor=torch.tensor([[1.,2.,3.,4.,5.,6.,7.,9.]])
        targets=torch.arange(24,dtype=torch.float32).reshape(1,3,8)
        encoded=encode_actions(anchor,targets)
        torch.testing.assert_close(encoded[...,7],targets[...,7])
        torch.testing.assert_close(decode_actions(anchor,encoded),targets)
        stats=ds.moments()
        q=np.arange(48,dtype=np.float64).reshape(6,8)
        expected=np.array([np.r_[q[j,:7]-q[i,:7],q[j,7]]
                           for i in range(5) for j in range(i+1,6)])
        np.testing.assert_allclose(stats['action_mean'],expected.mean(0))
        np.testing.assert_allclose(stats['action_std'],expected.std(0))
        normalized=ds[4]['action']
        physical=normalized*torch.tensor(stats['action_std'])+torch.tensor(stats['action_mean'])
        torch.testing.assert_close(physical,last['action'])
        print('Eight-task data, relative arm/absolute gripper, statistics, roundtrip and padding passed',flush=True)
if __name__=='__main__':main()
