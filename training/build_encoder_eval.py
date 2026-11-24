"""Freeze diverse four-frame raw sensor clips from the eight policy tasks.

One clip contains one sensor, four actual stored frames, stride two. Val/test
have disjoint source episodes. These episodes ARE part of policy training;
they are held out only relative to contact-only encoder pretraining.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from encoder.tactile import sensor_group, representation_spec
from training.data import TASKS, decode


def candidates(files, side):
    result=[]
    for path in files:
        with h5py.File(path) as f:
            group=sensor_group(f,side)
            markers=group['marker_full' if 'marker_full' in group else 'marker'][:]
            if len(markers)<7:
                continue
            displacement=markers[:,1]-markers[:,0]
            indices=np.arange(len(markers)-6)[:,None]+np.array([0,2,4,6])[None]
            clips=displacement[indices]
            change=np.linalg.norm(np.diff(clips,axis=1),axis=-1).mean((1,2))
            best=int(change.argmax())
            d=clips[best]
            magnitude=np.linalg.norm(d,axis=-1)
            features=np.r_[d.mean((0,1)),d.std((0,1)),np.quantile(magnitude,[.5,.9,.99]),change[best]]
            result.append(dict(source=str(path),side=side,indices=indices[best].tolist(),
                               temporal_change_px=float(change[best]),features=features.tolist()))
    return result


def diverse_select(rows,count,excluded):
    rows=[r for r in rows if r['source'] not in excluded]
    rows.sort(key=lambda r:r['temporal_change_px'],reverse=True)
    # Favor visible temporal change, then cover different displacement patterns.
    rows=rows[:max(count, int(len(rows)*.65))]
    if len(rows)<count:
        raise ValueError('Too few distinct episodes for balanced diverse clips')
    x=np.array([r['features'] for r in rows])
    x=(x-x.mean(0))/np.maximum(x.std(0),1e-4)
    chosen=[0]
    while len(chosen)<count:
        distance=((x[:,None]-x[chosen][None])**2).sum(-1).min(-1)
        distance[chosen]=-1
        chosen.append(int(distance.argmax()))
    return [rows[i] for i in chosen]


def build(root,output,per_task=20):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    archive=output/'raw_clips.h5'
    if archive.exists():
        raise FileExistsError(f'Refusing to overwrite frozen archive: {archive}')
    if per_task%4:
        raise ValueError('per_task must be divisible by four for side/split balance')
    manifest=[]
    with h5py.File(archive,'w') as target:
        target.attrs.update(schema='raw_tactile_eval_v1',frames=4,stride=2,
                            note='Encoder holdout from policy tasks; overlaps policy training, disjoint from contact pretraining')
        for task in TASKS:
            files=sorted((Path(root)/task/'hdf5').glob('*.hdf5'))
            selected=[];excluded=set()
            for side in ('left','right'):
                rows=diverse_select(candidates(files,side),per_task//2,excluded)
                excluded.update(r['source'] for r in rows);selected.extend(rows)
            for i,row in enumerate(selected):
                split='val' if i%2==0 else 'test'
                key=f'{split}/{task}_{i:03d}_{row["side"]}'
                out=target.create_group(key)
                with h5py.File(row['source']) as source:
                    g=sensor_group(source,row['side'])
                    mk='marker_full' if 'marker_full' in g else 'marker'
                    idx=row['indices']
                    depth=g['depth'][idx];markers=g[mk][idx]
                    out.create_dataset('depth',data=depth,compression='gzip',shuffle=True)
                    out.create_dataset('marker',data=markers,compression='gzip',shuffle=True)
                    rgb=out.create_dataset('rgb_marker',(4,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
                    digest=hashlib.sha256(depth.tobytes()+markers.tobytes())
                    for j,t in enumerate(idx):
                        raw=g['rgb_marker'][t]
                        if isinstance(raw,(bytes,np.bytes_)):
                            value=np.frombuffer(raw,dtype=np.uint8)
                        else:
                            # Current archives use encoded bytes. Keep raw array
                            # shape explicitly if a future source stores RGB arrays.
                            value=np.asarray(raw,dtype=np.uint8).reshape(-1)
                            out.attrs['rgb_array_shape']=g['rgb_marker'].shape[1:]
                        rgb[j]=value;digest.update(value.tobytes())
                        if 'rgb_array_shape' not in out.attrs:
                            decode(value)  # Fail before publishing corrupt raw RGB.
                    out.attrs.update(source=row['source'],side=row['side'],task=task,
                                     indices=idx,split=split,content_sha256=digest.hexdigest())
                row.update(key=key,task=task,split=split,content_sha256=digest.hexdigest())
                manifest.append(row)
            print(f'{task}: saved {len(selected)} clips from distinct episodes',flush=True)
    summary=dict(schema='raw_tactile_eval_v1',archive=archive.name,seed=42,frames_per_clip=4,stride=2,
                 clips=len(manifest),sensor_frames=4*len(manifest),per_task=per_task,
                 splits={s:sum(r['split']==s for r in manifest) for s in ('val','test')},
                 selection='Each episode: maximum mean consecutive sampled marker-displacement change; top 65% motion pool; greedy farthest-point normalized motion descriptors; balanced sensors; distinct source episode per task.',
                 representation=representation_spec(),
                 note='Raw RGB, camera depth and paired marker coordinates are copied, not externally linked. Conversion occurs at evaluation. Val/test episodes are disjoint; both are held out from contact encoder training, not from policy training.',
                 archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),samples=manifest)
    (output/'manifest.json').write_text(json.dumps(summary,indent=2))
    for split in ('val','test'):
        (output/f'{split}.json').write_text(json.dumps([r['key'] for r in manifest if r['split']==split],indent=2))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/yichi_data/UniVTAC_downloads/data/isaac51')
    p.add_argument('--output',default='data/encoder_eval_v1')
    p.add_argument('--per-task',type=int,default=20)
    args=p.parse_args();summary=build(args.root,args.output,args.per_task)
    print(json.dumps({k:v for k,v in summary.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
