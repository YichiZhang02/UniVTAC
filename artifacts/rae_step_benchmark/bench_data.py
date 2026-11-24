import json,time,bisect
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader,Subset
from threadpoolctl import threadpool_limits
from training.encoder_data import Contact,initialize_worker

class Cached(Dataset):
 def __init__(self,path):self.path=path;self.data=None
 def __len__(self):return 512
 def __getitem__(self,i):
  if self.data is None:self.data=np.load(self.path,mmap_mode='r')
  return torch.from_numpy(np.array(self.data[i],copy=True))

def main():
 torch.set_num_threads(2)
 out=Path('artifacts/rae_step_benchmark');out.mkdir(parents=True,exist_ok=True)
 path='/tmp/rae_radial_cache_benchmark.npy'
 with threadpool_limits(limits=1):
  ds=Contact('/mnt/yichi_data/UniVTAC_downloads/data/contact')
  indices=np.random.default_rng(73).integers(0,len(ds),512).tolist()
  raw=Subset(ds,indices)
  mmap=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(512,4,3,224,224))
  start=time.perf_counter()
  for i,x in enumerate(DataLoader(raw,batch_size=4,num_workers=4,worker_init_fn=initialize_worker)):
   mmap[i*4:(i+1)*4]=x.numpy()
  mmap.flush();del mmap
  print('Cache prepared seconds',time.perf_counter()-start,flush=True)
  results=[]
  for mode,data in [('online',raw),('precomputed_fp32',Cached(path))]:
   for workers in (4,8):
    loader=DataLoader(data,batch_size=64,num_workers=workers,worker_init_fn=initialize_worker,pin_memory=True,persistent_workers=True)
    for repeat in range(2):
     times=[];start=time.perf_counter();previous=start
     for x in loader:
      now=time.perf_counter();times.append(now-previous);previous=now
     elapsed=time.perf_counter()-start
     result=dict(mode=mode,workers=workers,repeat=repeat,batch=64,clips=512,total_s=elapsed,mean_s_batch=elapsed/8,median_next_wait_s=float(np.median(times)),first_batch_wait_s=times[0])
     results.append(result);print(json.dumps(result),flush=True)
    del loader
  # Host-to-device transfer on the same A100 used for compute measurements.
  batch=next(iter(DataLoader(Cached(path),batch_size=64,num_workers=0,pin_memory=True)))
  torch.cuda.set_device(1);events=[]
  for i in range(12):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();gpu=batch.to('cuda:1',non_blocking=True);b.record();b.synchronize()
   events.append(a.elapsed_time(b));del gpu
  report=dict(note='512 sampled real radial clips; precomputed FP32 cache fits page cache. Loader timings include HDF5/conversion or mmap/collation/IPC/pinning; no forward/backward. Cached layout stores four frames contiguously per clip for this benchmark; full production should store each frame once.',results=results,H2D_fp32_64clips_ms=float(np.median(events[2:])),full_unique_frame_cache_GiB=dict(fp32=len(ds)*3*224*224*4/2**30,fp16=len(ds)*3*224*224*2/2**30))
  (out/'data.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)
  ds.close()

if __name__=='__main__':main()
