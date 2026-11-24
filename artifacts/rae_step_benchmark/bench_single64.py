import gc,json,time
from pathlib import Path
import numpy as np,torch
from encoder.rae import RAE

torch.set_num_threads(2)
device='cuda:1';torch.cuda.set_device(device)
results=[]
for size in ('S','B'):
 for variant in ('rae-1in1','rae-4in1'):
  torch.manual_seed(42)
  model=RAE(variant=variant,size=size).to(device).train()
  opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.05)
  x=torch.rand(64,4,3,224,224,device=device)
  for _ in range(3):
   opt.zero_grad(set_to_none=True)
   for _ in range(1):
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=(model(x)-x).square().mean()
    loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
  timings=[]
  for update in range(12):
   events=[];wall=time.perf_counter();opt.zero_grad(set_to_none=True)
   for _ in range(1):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True)
    a.record()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=(model(x)-x).square().mean()
    b.record();loss.backward();c.record();events.append((a,b,c))
   d=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
   d.record();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();e.record()
   torch.cuda.synchronize()
   timings.append(dict(forward_ms=sum(a.elapsed_time(b) for a,b,c in events),backward_ms=sum(b.elapsed_time(c) for a,b,c in events),clip_optimizer_ms=d.elapsed_time(e),wall_ms=(time.perf_counter()-wall)*1000))
  result=dict(size=size,variant=variant,gpu=torch.cuda.get_device_name(device),micro_batch=64,accumulation=1,
              local_clips_per_update=64,global_clips_at_4ranks=64,note='Single-GPU compute only, data already on GPU; no DDP communication or data wait',
              timings={k:float(np.median([t[k] for t in timings])) for k in timings[0]},peak_GiB=torch.cuda.max_memory_allocated(device)/2**30)
  results.append(result);print(json.dumps(result),flush=True)
  del model,opt,x,loss;gc.collect();torch.cuda.empty_cache()
Path('artifacts/rae_step_benchmark').mkdir(parents=True,exist_ok=True)
Path('artifacts/rae_step_benchmark/gpu_single64.json').write_text(json.dumps(results,indent=2))
