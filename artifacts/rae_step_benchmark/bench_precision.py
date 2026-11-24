import contextlib,gc,json,time
from pathlib import Path
import numpy as np,torch
from encoder.rae import RAE

def main():
 torch.set_num_threads(2);torch.cuda.set_device(1);device='cuda:1'
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 results=[]
 for size in ('S','B'):
  for variant in ('rae-1in1','rae-4in1'):
   for precision in ('fp32','amp_bf16','bf16'):
    torch.manual_seed(42)
    model=RAE(variant=variant,size=size).to(device)
    if precision=='bf16':model.to(torch.bfloat16)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.05)
    target=torch.rand(64,4,3,224,224,device=device)
    x=target.to(torch.bfloat16) if precision=='bf16' else target
    torch.cuda.reset_peak_memory_stats(device);times=[];losses=[]
    for step in range(9):
     a,b,c,d=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
     torch.cuda.synchronize();wall=time.perf_counter();opt.zero_grad(set_to_none=True);a.record()
     with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='amp_bf16'):
      prediction=model(x);loss=(prediction.float()-target).square().mean()
     b.record();loss.backward();c.record()
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite precision benchmark')
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();d.record();torch.cuda.synchronize()
     losses.append(loss.item())
     if step>=3:times.append(dict(forward_ms=a.elapsed_time(b),backward_ms=b.elapsed_time(c),optimizer_ms=c.elapsed_time(d),wall_ms=(time.perf_counter()-wall)*1000))
    result=dict(size=size,variant=variant,precision=precision,gpu=torch.cuda.get_device_name(1),batch=64,
                timings={k:float(np.median([row[k] for row in times])) for k in times[0]},
                peak_GiB=torch.cuda.max_memory_allocated(1)/2**30,initial_loss=losses[0],final_loss=losses[-1],
                parameters=str(next(model.parameters()).dtype),
                optimizer_state=str(next(iter(opt.state.values()))['exp_avg'].dtype),
                note='Single A100; synthetic fixed inputs; six measured full updates after three warmups; native BF16 has BF16 AdamW moments, FP32 loss; no convergence claim')
    print(json.dumps(result),flush=True);results.append(result)
    Path('artifacts/rae_step_benchmark/precision.json').write_text(json.dumps(results,indent=2))
    del model,opt,target,x,loss,prediction;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
