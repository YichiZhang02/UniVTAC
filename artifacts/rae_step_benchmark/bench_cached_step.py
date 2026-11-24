import gc,json,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from training.encoder_data import initialize_worker
from encoder.rae import RAE

class Cached(Dataset):
 def __init__(self):self.data=None
 def __len__(self):return 512
 def __getitem__(self,i):
  if self.data is None:self.data=np.load('/tmp/rae_radial_cache_benchmark.npy',mmap_mode='r')
  return torch.from_numpy(np.array(self.data[i],copy=True))

def main():
 torch.set_num_threads(2);torch.cuda.set_device(1);device='cuda:1';results=[]
 for size in ('S','B'):
  for variant in ('rae-1in1','rae-4in1'):
   model=RAE(variant=variant,size=size).to(device).train();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.05)
   loader=DataLoader(Cached(),batch_size=64,num_workers=4,worker_init_fn=initialize_worker,pin_memory=True,persistent_workers=True)
   samples=[]
   for repeat in range(2):
    it=iter(loader)
    for batch_index in range(8):
     torch.cuda.synchronize();start=time.perf_counter();cpu=next(it);data_end=time.perf_counter()
     x=cpu.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.bfloat16):loss=(model(x)-x).square().mean()
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
     elapsed=time.perf_counter()-start
     if repeat==1 and batch_index>=2:samples.append(dict(wall_s=elapsed,data_wait_s=data_end-start))
   result=dict(size=size,variant=variant,gpu=torch.cuda.get_device_name(1),batch=64,
               wall_s_median=float(np.median([v['wall_s'] for v in samples])),wall_s_mean=float(np.mean([v['wall_s'] for v in samples])),
               data_wait_s_mean=float(np.mean([v['data_wait_s'] for v in samples])),
               note='End-to-end single GPU, cached FP32 radial tensors in page cache, 4 workers; six steady-state updates excluding startup/warmup')
   results.append(result);print(json.dumps(result),flush=True)
   del model,opt,loader,it,cpu,x,loss;gc.collect();torch.cuda.empty_cache()
 Path('artifacts/rae_step_benchmark/cached_end_to_end.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
