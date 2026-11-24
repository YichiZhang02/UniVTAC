"""Reproducible DDP co-training; steps count optimizer updates, never micro-batches."""
from training.tactile_vlm import add_tactile_arguments, apply_tactile_arguments
import argparse
import contextlib
import hashlib
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from training.data import Episodes
from training.actions import ACTION_CONTRACT
from training.encoder_data import initialize_worker
from training.models import ACT,StarVLA
from policy.starvla_pi import StarVLAPI
from policy.starvla_groot_v1 import StarVLAGrootV1
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    add_tactile_arguments(p)
    p.add_argument('--model',choices=['act','starvla','starvla_pi','starvla_groot_v1'],required=True)
    p.add_argument('--config',default=str(ROOT/'training/config.json'))
    p.add_argument('--output',required=True)
    p.add_argument('--resume',help='Full model and optimizer checkpoint')
    p.add_argument('--steps',type=int)
    p.add_argument('--micro-batch',type=int)
    p.add_argument('--tactile',action='store_true')
    p.add_argument('--encoder',choices=['univtac','rae'])
    p.add_argument('--rae-variant',choices=['rae-1in1','rae-4in1'])
    p.add_argument('--encoder-checkpoint')
    p.add_argument('--rae-size',choices=['S','B'])
    p.add_argument('--tokens',choices=['full','pool3x3','learned3x3','cls'])
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());apply_tactile_arguments(args,cfg)
    if args.steps is not None: cfg['steps']=args.steps
    if args.micro_batch is not None: cfg['micro_batch']=args.micro_batch
    if args.tactile: cfg['tactile']=True
    if args.encoder: cfg['encoder']=args.encoder
    if args.rae_variant: cfg['rae_variant']=args.rae_variant
    if args.encoder_checkpoint: cfg['encoder_checkpoint']=args.encoder_checkpoint
    if args.rae_size: cfg['rae_size']=args.rae_size
    if cfg['encoder']=='rae':
        cfg['tactile_frames']=1 if cfg.get('rae_variant')=='rae-1in1' else 4
    if args.tokens: cfg['tactile_tokens']=args.tokens
    if args.model=='starvla_groot_v1':cfg.update(head_action_queries_only=False,action_future_tokens=0)
    rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    torch.cuda.set_device(local)
    if world>1: dist.init_process_group('nccl')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    denominator=world*cfg['micro_batch']
    if cfg['global_batch']%denominator:
        raise ValueError(f"global_batch must be divisible by world_size*micro_batch ({denominator})")
    accum=cfg['global_batch']//denominator
    ds=Episodes(cfg['data_root'],ROOT/'instructions',cfg['chunk'],cfg['tactile'],
                cfg['tactile_frames'],cfg['tactile_stride'],
                cfg.get('tactile_input_mode','depth_deform_radial'),
                cfg.get('tactile_interpolation','tps_bilinear'),cfg.get('tactile_interpolation_grid',28))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        versions={}
        for package in ('torch','torchvision','transformers','diffusers','modelscope','omegaconf','numpy','h5py'):
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]=None
        versions.update(python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(local))
        (output/'environment.json').write_text(json.dumps(versions,indent=2))
        stats=ds.moments()
        (output/'stats.json').write_text(json.dumps(stats,indent=2))
        files=[{'path':str(f),'bytes':f.stat().st_size} for f in ds.files]
        (output/'train_manifest.json').write_text(json.dumps(files,indent=2))
        cfg.update(model=args.model,world_size=world,gradient_accumulation_steps=accum,
                   train_episodes=len(ds.files),train_frames=len(ds),test_episodes=0,
                   action_contract=ACTION_CONTRACT,
                   relative_joint_definition='arm: q[t+k+1,:7] - q[t,:7]; gripper: q[t+k+1,7]; terminal hold padding',
                   source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                  for folder in ('training','encoder') for f in (ROOT/folder).glob('*.py')})
        (output/'config.json').write_text(json.dumps(cfg,indent=2))
    if world>1: dist.barrier()
    ds.stats=json.loads((output/'stats.json').read_text())
    sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=cfg['seed'])
    loader=DataLoader(ds,batch_size=cfg['micro_batch'],sampler=sampler,num_workers=args.workers,
                      pin_memory=True,drop_last=True,worker_init_fn=initialize_worker)
    if not len(loader): raise ValueError('Dataset smaller than a local batch')
    model=({'act': ACT, 'starvla': StarVLA, 'starvla_pi': StarVLAPI, 'starvla_groot_v1': StarVLAGrootV1}[args.model])(cfg,len(ds.stats['state_mean'])).cuda()
    if args.model in ('starvla','starvla_pi','starvla_groot_v1'):
        vlm=list(model.native.qwen_vl_interface.parameters())
        vlm_ids={id(p) for p in vlm}
        other=[p for p in model.parameters() if id(p) not in vlm_ids]
        action_lr=cfg['learning_rate'] if cfg.get('lr_schedule','cosine')=='cosine' else cfg.get('action_learning_rate',1e-4)
        groups=[dict(params=vlm,lr=cfg['learning_rate']),dict(params=other,lr=action_lr)]
        opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
    from training.lr_schedule import build
    scheduler=build(opt,cfg) if args.model in ('starvla','starvla_pi','starvla_groot_v1') else None
    saved=None
    if args.resume:
        from training.resume_checkpoint import restore
        resume_cfg=dict(cfg,model=args.model,world_size=world)
        saved=restore(args.resume,model,opt,resume_cfg,ds.stats)
        if scheduler is not None:
            if saved.get('scheduler') is None:raise ValueError('Scheduled training requires scheduler state to resume')
            scheduler.load_state_dict(saved['scheduler'])
    if world>1: model=DDP(model,device_ids=[local],find_unused_parameters=True)
    epoch=0;sampler.set_epoch(epoch);iterator=iter(loader)
    start_step=saved['step'] if saved else 0
    if saved:
        # Position the sampler without rereading discarded training batches.
        consumed=start_step*accum
        epoch,offset=divmod(consumed,len(loader))
        sampler.set_epoch(epoch)
        indices=list(iter(sampler))[offset*cfg['micro_batch']:]
        loader_tail=DataLoader(ds,batch_size=cfg['micro_batch'],sampler=indices,num_workers=args.workers,
                               pin_memory=True,drop_last=True,worker_init_fn=initialize_worker)
        iterator=iter(loader_tail)
        from training.resume_checkpoint import restore_rng
        restore_rng(saved)
        del saved
    for step in range(start_step+1,cfg['steps']+1):
        opt.zero_grad(set_to_none=True);total=0.
        for micro in range(accum):
            try: batch=next(iterator)
            except StopIteration:
                epoch+=1;sampler.set_epoch(epoch);iterator=iter(loader);batch=next(iterator)
            batch={k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            sync=model.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(batch);scaled=loss/accum
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {step}')
            scaled.backward();total+=float(loss.detach())/accum
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        used_lrs=[group['lr'] for group in opt.param_groups]
        opt.step()
        if scheduler is not None:scheduler.step()
        if rank==0 and (step==1 or step%10==0):
            record={'step':step,'loss':total,'epoch':epoch,'global_batch':cfg['global_batch'],'learning_rates':used_lrs}
            print(json.dumps(record),flush=True)
            with (output/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if step%cfg['save_interval']==0 or step==cfg['steps']:
            if rank==0:
                native=model.module if world>1 else model
                target=output/f'step_{step:06d}.pt';temp=target.with_suffix('.tmp')
                torch.save(dict(model=native.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,step=step,
                                config=cfg,stats=ds.stats,rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all()),temp)
                temp.replace(target)
            if world>1:dist.barrier()
    if world>1: dist.destroy_process_group()
if __name__=='__main__':main()
