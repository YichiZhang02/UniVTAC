"""Verify frozen archive contents, split isolation and recorded/live conversion."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from encoder.tactile import from_group,from_observation,sensor_group
from training.data import TASKS,decode
from training.encoder_data import Contact


def main():
    root=Path('data/encoder_eval_v1')
    manifest=json.loads((root/'manifest.json').read_text())
    rows=manifest['samples']
    assert len(rows)==160 and Counter(r['task'] for r in rows)==Counter({t:20 for t in TASKS})
    assert Counter(r['side'] for r in rows)==Counter(left=80,right=80)
    assert Counter(r['split'] for r in rows)==Counter(val=80,test=80)
    episodes={s:{r['source'] for r in rows if r['split']==s} for s in ('val','test')}
    assert not episodes['val'] & episodes['test']
    with (root/'raw_clips.h5').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['archive_sha256']
    preview=[];ranges=[]
    with threadpool_limits(limits=1),h5py.File(root/'raw_clips.h5') as archive:
        for row in rows:
            assert np.diff(row['indices']).tolist()==[2,2,2]
            saved=archive[row['key']]
            with h5py.File(row['source']) as src:
                source=sensor_group(src,row['side'])
                mk='marker_full' if 'marker_full' in source else 'marker'
                digest=hashlib.sha256(saved['depth'][:].tobytes()+saved['marker'][:].tobytes())
                converted=[]
                for i,t in enumerate(row['indices']):
                    raw=saved['rgb_marker'][i].tobytes()
                    assert raw==source['rgb_marker'][t].tobytes()
                    digest.update(raw)
                    np.testing.assert_array_equal(saved['depth'][i],source['depth'][t])
                    np.testing.assert_array_equal(saved['marker'][i],source[mk][t])
                    offline=from_group(saved,i)
                    online=from_observation({'depth':torch.from_numpy(saved['depth'][i]),'marker':torch.from_numpy(saved['marker'][i])})
                    torch.testing.assert_close(offline,online,rtol=0,atol=0)
                    assert offline.shape==(3,224,224) and torch.isfinite(offline).all()
                    assert offline.min()>=0 and offline.max()<=1
                    converted.append(offline.permute(1,2,0).numpy())
                assert digest.hexdigest()==row['content_sha256']
            if row['task'] not in [r[0] for r in preview]:
                preview.append((row['task'],decode(saved['rgb_marker'][0]),converted))
            ranges.append(row['temporal_change_px'])
        contact=Contact('/mnt/yichi_data/UniVTAC_downloads/data/contact')
        assert len(contact.files)==638 and len(contact.items)==1276
        train_paths={str(p) for p in contact.files}
        assert not train_paths & (episodes['val']|episodes['test'])
        for index in (0,len(contact)//2,len(contact)-1):
            x=contact[index];assert x.shape==(4,3,224,224) and torch.isfinite(x).all()
        report=dict(clips=160,sensor_frames=640,per_task=20,stride=2,val_clips=80,test_clips=80,
                    contact_files=len(contact.files),contact_sensor_anchors=len(contact),
                    val_test_source_episodes_disjoint=True,contact_eval_sources_disjoint=True,
                    all_raw_rgb_depth_markers_match_sources=True,all_offline_online_conversions_bitwise_equal=True,
                    temporal_marker_change_px=dict(min=float(min(ranges)),median=float(np.median(ranges)),max=float(max(ranges))))
        contact.close()
    (root/'verification.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(8,5,figsize=(12,18))
    for axrow,(task,rgb,converted) in zip(axes,preview):
        for ax,image in zip(axrow,[rgb]+converted):ax.imshow(image);ax.set_xticks([]);ax.set_yticks([])
        axrow[0].set_ylabel(task)
    for ax,title in zip(axes[0],['Original marker RGB','Radial t','Radial t+2','Radial t+4','Radial t+6']):ax.set_title(title)
    fig.tight_layout();fig.savefig(root/'preview.png',dpi=140);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
