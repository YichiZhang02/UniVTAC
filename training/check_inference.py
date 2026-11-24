import torch
from training.inference import ChunkPolicy
p=ChunkPolicy('/mnt/yichi_data/UniVTAC_downloads/local-ddp-smoke-act/step_000001.pt')
b=dict(state=torch.zeros(1,8),image=torch.zeros(1,2,3,224,224),tactile=torch.empty(1,0),lang=['Lift the can.'])
a=p.predict(b)
assert a.shape==(1,30,8) and torch.isfinite(a).all()
print('ACT checkpoint reload and 30-step absolute inference passed',flush=True)
