import json
from pathlib import Path
import h5py
import numpy as np
from training.data import TASKS
root=Path('/tmp/univtac-synthetic-data');root.mkdir(exist_ok=True)
for task in TASKS:
 d=root/task/'hdf5';d.mkdir(parents=True,exist_ok=True)
 with h5py.File(d/'0.hdf5','w') as f:
  f['embodiment/joint']=np.random.default_rng(42).normal(size=(20,9)).astype('float32')
  for cam in ('head','wrist'): f[f'observation/{cam}/rgb']=np.zeros((20,24,24,3),np.uint8)
cfg=json.loads(Path('training/config.json').read_text());cfg['data_root']=str(root)
Path('/tmp/univtac-smoke-config.json').write_text(json.dumps(cfg))
