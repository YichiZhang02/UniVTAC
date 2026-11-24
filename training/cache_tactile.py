"""Precompute lossless FP32 radial episodes and the frozen encoder holdouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from encoder.tactile import from_group,representation_spec,sensor_group
from training.encoder_data import initialize_worker


def atomic(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def convert_episode(job):
    source,relative,output,method,grid=job
    source=Path(source);output=Path(output);spec=representation_spec(method,grid)
    folder=output/Path(relative).parent.parent/Path(relative).stem
    folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/'metadata.json';stat=source.stat()
    identity=dict(path=str(source),relative=relative,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if metadata.exists():
        old=json.loads(metadata.read_text())
        if old['source']==identity and old['representation']==spec:
            good=True
            for sensor in old['sensors']:
                path=output/sensor['path']
                if not path.exists():good=False;break
                data=np.load(path,mmap_mode='r')
                if list(data.shape)!=sensor['shape'] or data.dtype!=np.float32:good=False;break
            if good:return old
        raise RuntimeError(f'Existing cache metadata differs: {metadata}')
    sensors=[]
    with h5py.File(source,'r') as f:
        for side in ('left','right'):
            group=sensor_group(f,side)
            key='marker_full' if 'marker_full' in group else 'marker'
            # Sequential episode reads replace frame-wise HDF5 metadata/I/O.
            depths=group['depth'][:];markers=group[key][:]
            if len(depths)!=len(markers):raise ValueError(f'Misaligned arrays: {source}/{side}')
            target=folder/f'{side}.npy';temp=folder/f'{side}.tmp.npy'
            shape=[len(depths),3,224,224]
            digest=hashlib.sha256();minimum=1.;maximum=0.
            from encoder.tactile import convert
            # Sequential buffered writes avoid NFS mmap write-fault read traffic.
            with temp.open('wb',buffering=8*1024*1024) as stream:
                np.lib.format.write_array_header_2_0(stream,dict(descr=np.dtype('float32').str,
                                                               fortran_order=False,shape=tuple(shape)))
                for t in range(len(depths)):
                    value=convert(depths[t],markers[t],method=method,grid=grid).numpy()
                    if not np.isfinite(value).all():raise ValueError(f'Nonfinite output: {source}/{side}/{t}')
                    minimum=min(minimum,float(value.min()));maximum=max(maximum,float(value.max()))
                    payload=value.tobytes();stream.write(payload);digest.update(payload)
                stream.flush();os.fsync(stream.fileno())
            temp.replace(target)
            sensors.append(dict(side=side,path=str(target.relative_to(output)),frames=len(depths),shape=shape,
                                dtype='float32',payload_sha256=digest.hexdigest(),min=minimum,max=maximum,
                                bytes=target.stat().st_size))
            del depths,markers
    after=source.stat()
    if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):raise RuntimeError(f'Source changed: {source}')
    result=dict(source=identity,representation=spec,sensors=sensors)
    atomic(metadata,result);return result


def convert_holdouts(root,method,grid):
    root=Path(root);raw=root/'raw_clips.h5';source=json.loads((root/'manifest.json').read_text())
    spec=representation_spec(method,grid);target=root/'depth_deform_radial.npy';metadata=root/'depth_deform_radial.json'
    if metadata.exists():
        old=json.loads(metadata.read_text())
        if old['representation']==spec and old['raw_archive_sha256']==source['archive_sha256'] and target.exists():return old
        raise RuntimeError('Frozen holdout cache differs; use a new cache version')
    keys=[r['key'] for r in source['samples']]
    temp=root/'depth_deform_radial.tmp.npy'
    data=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(keys),4,3,224,224))
    digest=hashlib.sha256()
    with raw.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=source['archive_sha256']:raise ValueError('Raw holdout checksum mismatch')
    with h5py.File(raw,'r') as f:
        for i,key in enumerate(keys):
            clip=torch.stack([from_group(f[key],t,method=method,grid=grid) for t in range(4)]).numpy()
            data[i]=clip;digest.update(clip.tobytes())
    data.flush();shape=list(data.shape);del data;temp.replace(target)
    report=dict(schema='cached_radial_holdouts_v1',complete=True,representation=spec,path=target.name,
                shape=shape,dtype='float32',keys=keys,clips=len(keys),sensor_frames=4*len(keys),
                splits={s:sum(k.startswith(s+'/') for k in keys) for s in ('val','test')},
                raw_archive_sha256=source['archive_sha256'],payload_sha256=digest.hexdigest(),bytes=target.stat().st_size)
    atomic(metadata,report);return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/yichi_data/UniVTAC_downloads/data/contact')
    p.add_argument('--output',default='data/contact_depth_deform_radial_v1')
    p.add_argument('--eval-root',default='data/encoder_eval_v1')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();initialize_worker(0);torch.set_num_threads(1)
    root=Path(args.root).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    files=sorted(root.glob('*/hdf5/*.hdf5'))
    if not files:raise ValueError('No contact episodes')
    start=time.time();rows=[]
    holdout=convert_holdouts(args.eval_root,'tps_bilinear',28)
    print(json.dumps(dict(event='holdouts_complete',clips=holdout['clips'],sensor_frames=holdout['sensor_frames'])),flush=True)
    jobs=[(str(f),str(f.relative_to(root)),str(output),'tps_bilinear',28) for f in files]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(0,)) as pool:
        futures=[pool.submit(convert_episode,job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            report=dict(status='Converting',episodes_done=len(rows),episodes_total=len(files),
                        frames_done=sum(s['frames'] for row in rows for s in row['sensors']),elapsed_s=time.time()-start)
            atomic(output/'progress.json',report)
            if len(rows)%10==0 or len(rows)==len(files):print(json.dumps(report),flush=True)
    rows.sort(key=lambda r:r['source']['relative'])
    manifest=dict(schema='cached_radial_contact_v1',complete=True,dtype='float32',layout='NCHW',
                  representation=representation_spec(),source_root=str(root),episodes=rows,episode_count=len(rows),
                  sensor_streams=sum(len(r['sensors']) for r in rows),
                  sensor_frames=sum(s['frames'] for r in rows for s in r['sensors']),
                  bytes=sum(s['bytes'] for r in rows for s in r['sensors']),
                  note='One cached tensor per original sensor frame; chronological stride-2 histories assembled at training time. No RGB/depth/marker source data modified.')
    atomic(output/'manifest.json',manifest)
    atomic(output/'progress.json',dict(status='Completed',episodes_done=len(rows),episodes_total=len(files),
                                     sensor_frames=manifest['sensor_frames'],elapsed_s=time.time()-start))
    print(json.dumps({k:v for k,v in manifest.items() if k!='episodes'},indent=2),flush=True)


if __name__=='__main__':main()
