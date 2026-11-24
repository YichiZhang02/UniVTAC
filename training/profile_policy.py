"""Short, isolated PI training profile; never saves or resumes policy weights."""
import argparse
import contextlib
import functools
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from training.data import Episodes
from training.encoder_data import initialize_worker
from policy.starvla_pi import StarVLAPI


def annotate(obj,name,label):
    original=getattr(obj,name)
    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        with torch.profiler.record_function(label):return original(*args,**kwargs)
    setattr(obj,name,wrapped)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--stats',required=True)
    p.add_argument('--output',required=True);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--measure',type=int,default=8);p.add_argument('--trace',action='store_true')
    p.add_argument('--cases',nargs='+',choices=['real','cached','cpu_inputs'],default=['real','cached','cpu_inputs'])
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl')
    torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    ds=Episodes(cfg['data_root'],root/'instructions',cfg['chunk'],cfg['tactile'],cfg['tactile_frames'],cfg['tactile_stride'],cfg['tactile_input_mode'],cfg['tactile_interpolation'],cfg['tactile_interpolation_grid'])
    ds.stats=json.loads(Path(a.stats).read_text())
    sampler=DistributedSampler(ds,num_replicas=world,rank=rank,seed=42)
    loader=DataLoader(ds,batch_size=cfg['micro_batch'],sampler=sampler,num_workers=4,pin_memory=True,drop_last=True,worker_init_fn=initialize_worker)
    iterator=iter(loader);cached=next(iterator)
    print(json.dumps(dict(event='data_ready',rank=rank,episodes=len(ds.files))),flush=True)
    model=StarVLAPI(cfg,len(ds.stats['state_mean'])).cuda().train()
    annotate(model,'encode','component.encode')
    annotate(model.tactile,'forward','component.rae')
    annotate(model.vlm_tactile,'run_text','component.qwen_text')
    interface=model.native.qwen_vl_interface
    annotate(interface,'build_qwenvl_inputs','component.cpu_processor')
    annotate(model.vlm_tactile,'forward','component.vlm_with_packing')
    annotate(model.native,'_project_vl_hidden_for_action','component.layer_projections')
    annotate(model.native.action_model,'forward','component.action_head')
    vlm=list(interface.parameters());ids={id(p) for p in vlm}
    opt=torch.optim.AdamW([dict(params=vlm,lr=cfg['learning_rate']),dict(params=[p for p in model.parameters() if id(p) not in ids],lr=cfg['learning_rate'])],weight_decay=1e-4)
    kernels={}
    for module in model.modules():
        if 'GatedDeltaNet' in type(module).__name__:
            kernels={k:str(getattr(module,k,None)) for k in ['chunk_gated_delta_rule','causal_conv1d_fn']};break
    if world>1:model=DDP(model,device_ids=[local],find_unused_parameters=True)
    accum=cfg['global_batch']//(world*cfg['micro_batch'])
    report=dict(config=cfg,world_size=world,accumulation=accum,gpu=torch.cuda.get_device_name(),torch=torch.__version__,kernels=kernels,cases={})
    print(json.dumps(dict(event='model_ready',rank=rank,kernels=kernels,accum=accum)),flush=True)

    def step(case):
        nonlocal iterator
        opt.zero_grad(set_to_none=True);wait=0.;total=0.
        for micro in range(accum):
            with torch.profiler.record_function('phase.next_batch'):
                begin=time.perf_counter()
                if case=='real':
                    try:batch=next(iterator)
                    except StopIteration:iterator=iter(loader);batch=next(iterator)
                else:batch=cached
                wait+=time.perf_counter()-begin
            with torch.profiler.record_function('phase.transfer'):
                batch={k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) and not (case=='cpu_inputs' and k in ('image','state')) else v for k,v in batch.items()}
            sync=model.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.profiler.record_function('phase.forward'):loss=model(batch);scaled=loss/accum
            with torch.profiler.record_function('phase.finite_check'):
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite profiling loss')
            with torch.profiler.record_function('phase.backward'):scaled.backward()
            with torch.profiler.record_function('phase.loss_item'):total+=float(loss.detach())/accum
        with torch.profiler.record_function('phase.clip'):torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        with torch.profiler.record_function('phase.adam'):opt.step()
        return wait,total

    for case in a.cases:
        for _ in range(a.warmup):step(case)
        durations=[];waits=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for _ in range(a.measure):
            begin=time.perf_counter();wait,loss=step(case);torch.cuda.synchronize()
            durations.append(time.perf_counter()-begin);waits.append(wait)
        report['cases'][case]=dict(seconds_mean=statistics.mean(durations),seconds_median=statistics.median(durations),data_wait_mean=statistics.mean(waits),durations=durations,loss=loss,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        print(json.dumps(dict(event='case_done',rank=rank,case=case,**report['cases'][case])),flush=True)
        (out/f'results.rank{rank}.json').write_text(json.dumps(report,indent=2))
    if a.trace and rank==0 and world==1:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            step('real');torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/'trace.json'))
        (out/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        rows=[]
        for event in prof.key_averages():
            if event.key.startswith(('component.','phase.')):
                rows.append(dict(name=event.key,calls=event.count,cpu_ms=event.cpu_time_total/1000,device_ms=event.device_time_total/1000))
        (out/'regions.json').write_text(json.dumps(rows,indent=2))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
