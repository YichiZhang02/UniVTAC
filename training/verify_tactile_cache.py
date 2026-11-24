"""Check all cached streams against raw first/middle/last frames and metadata."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import h5py
import numpy as np

from encoder.tactile import from_group,sensor_group
from training.encoder_data import initialize_worker
from training.cache_tactile import atomic


def verify_episode(job):
    root,row=job;root=Path(root);source=Path(row['source']['path'])
    stat=source.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(row['source']['bytes'],row['source']['mtime_ns']):
        raise ValueError(f'Raw source changed: {source}')
    frames_checked=0;total=0
    with h5py.File(source,'r') as f:
        for sensor in row['sensors']:
            path=root/sensor['path'];data=np.load(path,mmap_mode='r')
            if list(data.shape)!=sensor['shape'] or data.dtype!=np.float32:raise ValueError(f'Invalid shape/dtype: {path}')
            if path.stat().st_size!=sensor['bytes']:raise ValueError(f'Invalid cache size: {path}')
            if not 0<=sensor['min']<=sensor['max']<=1:raise ValueError(f'Invalid value range: {path}')
            group=sensor_group(f,sensor['side'])
            for t in sorted(set([0,len(data)//2,len(data)-1])):
                np.testing.assert_array_equal(data[t],from_group(group,t).numpy())
                frames_checked+=1
            total+=len(data)
    return dict(source=str(source),streams=len(row['sensors']),frames=total,checked=frames_checked)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/contact_depth_deform_radial_v1');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=Path(args.root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['complete'] and manifest['episode_count']==638 and manifest['sensor_frames']==353690
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(0,)) as pool:
        rows=list(pool.map(verify_episode,[(str(root),row) for row in manifest['episodes']]))
    report=dict(episodes_verified=len(rows),sensor_streams_verified=sum(r['streams'] for r in rows),
                sensor_frames=sum(r['frames'] for r in rows),raw_parity_frames=sum(r['checked'] for r in rows),
                first_middle_last_frames_bitwise_equal=True,all_shapes_dtypes_sizes_checked=True,
                all_frames_finite_and_in_0_1_at_conversion=True,source_identity_unchanged=True,storage_GiB=manifest['bytes']/2**30)
    atomic(root/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
