"""Contact-only RAE pretraining with fixed policy-task validation clips."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from encoder.rae import RAE, VARIANTS
from encoder.tactile import representation_spec
from training.encoder_data import Contact, FrozenClips, CachedContact, CachedFrozenClips, initialize_worker
from training.rae_checkpoint import architecture, save_best, save_checkpoint
from training.anchor_sampler import EpochAnchorSampler
from training.early_stop_rae import decision as early_stop_decision

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate(model, loader, device, precision='amp_bf16'):
    """Pixel-weighted MSE; single-frame RAE processes all four frames independently."""
    training = model.training
    model.eval()
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    pixels = 0
    for x in loader:
        x = x.to(device)
        inputs=x.to(torch.bfloat16) if precision=='bf16' else x
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda') and precision=='amp_bf16'):
            pred = model(inputs)
        error = (pred.float()-x.float()).square().double()
        sums += error.sum((0, 1, 3, 4))
        pixels += x.shape[0]*x.shape[1]*x.shape[3]*x.shape[4]
    model.train(training)
    if not pixels:
        raise ValueError('Empty evaluation split')
    channels = (sums/pixels).tolist()
    return dict(mse=sum(channels)/3, depth_mse=channels[0], dx_mse=channels[1], dy_mse=channels[2])


def train_epoch(model, loader, opt, device, *, accum, micro_batch, samples_per_rank,
                start_step=0, epoch=1, rank=0, world=1, log_path=None, precision='amp_bf16'):
    model.train(); step=start_step; epoch_sum=0.; count=0; update_loss=0.
    for batch_index,x in enumerate(loader):
        if batch_index%accum==0:
            opt.zero_grad(set_to_none=True);update_loss=0.
        group_start=(batch_index//accum)*accum*micro_batch
        group_samples=min(accum*micro_batch,samples_per_rank-group_start)
        last=(batch_index+1)%accum==0 or batch_index+1==len(loader)
        x=x.to(device,non_blocking=True)
        inputs=x.to(torch.bfloat16) if precision=='bf16' else x
        sync=model.no_sync() if world>1 and not last else contextlib.nullcontext()
        with sync,torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda') and precision=='amp_bf16'):
            loss=(model(inputs).float()-x.float()).square().mean()
            scaled=loss*(len(x)/group_samples)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        scaled.backward()
        value=float(loss.detach());epoch_sum+=value*len(x);count+=len(x)
        update_loss+=value*len(x)/group_samples
        if last:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
            if rank==0 and (step==1 or step%10==0):
                record=dict(step=step,epoch=epoch,mse=update_loss)
                print(json.dumps(record),flush=True)
                if log_path:
                    with Path(log_path).open('a') as f:f.write(json.dumps(record)+'\n')
    stats=torch.tensor([epoch_sum,count],device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(stats)
    return step,float(stats[0]/stats[1]),int(stats[1])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/yichi_data/UniVTAC_downloads/data/contact')
    p.add_argument('--data-mode',choices=['cached','online'],default='cached')
    p.add_argument('--cache-root',default=str(ROOT/'data/contact_depth_deform_radial_v1'))
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=10,help='Stop after this many completed epochs without a new best val MSE')
    p.add_argument('--samples-per-epoch',type=int,default=80000)
    p.add_argument('--variant',choices=VARIANTS,default='rae-4in1')
    p.add_argument('--size',choices=['S','B'],default='S')
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--interpolation',choices=['linear','tps_bilinear','tps'],default='tps_bilinear')
    p.add_argument('--grid',type=int,default=28)
    p.add_argument('--eval-root',default=str(ROOT/'data/encoder_eval_v1'))
    p.add_argument('--micro-batch',type=int,default=64,help='Four-frame clips per rank; default single-GPU batch64')
    p.add_argument('--global-batch',type=int,default=64,help='Four-frame clips per optimizer update')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--precision',choices=['amp_bf16','bf16','fp32'],default='amp_bf16')
    a=p.parse_args()
    if a.stride!=2:raise ValueError('Frozen evaluation uses four frames at stride 2')
    if min(a.epochs,a.micro_batch,a.global_batch,a.patience)<1:raise ValueError('Epochs, patience and batches must be positive')
    rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    if a.global_batch%(world*a.micro_batch):raise ValueError('global_batch must be divisible by world_size * micro_batch')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.manual_seed(42)
    if world>1:dist.init_process_group('nccl')
    ds=(CachedContact(a.cache_root,4,a.stride,a.interpolation,a.grid) if a.data_mode=='cached'
        else Contact(a.root,4,a.stride,a.interpolation,a.grid))
    sampler=EpochAnchorSampler(ds,a.samples_per_epoch,num_replicas=world,rank=rank,seed=42)
    loader=DataLoader(ds,batch_size=a.micro_batch,sampler=sampler,num_workers=a.workers,
                      pin_memory=True,drop_last=False,worker_init_fn=initialize_worker,
                      persistent_workers=a.workers>0)
    if not len(loader):raise ValueError('Empty training loader')
    holdout_class=CachedFrozenClips if a.data_mode=='cached' else FrozenClips
    val=holdout_class(a.eval_root,'val',a.interpolation,a.grid)
    # Materialize the fixed conversion once on rank zero, in CPU memory.
    val_loader=DataLoader([val[i] for i in range(len(val))],batch_size=1) if rank==0 else None
    val.close()
    model=RAE(variant=a.variant,size=a.size).to(device)
    if a.precision=='bf16':model.to(torch.bfloat16)
    spec=dict(variant=a.variant,size=a.size,frames=model.frames,latent_frames=model.encoder.latent_frames,
              input_mode='depth_deform_radial',position='learned_absolute',stride=a.stride,
              representation=representation_spec(a.interpolation,a.grid))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.05)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    resumed_epoch=0;resumed_step=0
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if saved['encoder_spec'] != spec:
            raise ValueError('Existing checkpoint does not match requested encoder specification')
        if saved.get('config',{}).get('precision','amp_bf16') != a.precision:
            raise ValueError('Resume precision differs; use the original precision or a new output directory')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        resumed_epoch=saved['epoch'];resumed_step=saved['step']
        del saved
    if world>1:model=DDP(model,device_ids=[local])
    accum=a.global_batch//(world*a.micro_batch)
    native=model.module if world>1 else model
    config=dict(vars(a),seed=42,world_size=world,encoder_spec=spec,architecture=architecture(native),
                train_episodes=len(ds.files),train_sensor_anchors=len(ds),
                sampler_samples_per_epoch=len(sampler)*world,
                unique_anchors_per_epoch=sampler.selected_count,
                sampler_padding_per_epoch=len(sampler)*world-sampler.selected_count,
                anchor_sampling='without replacement within epoch, resampled with seed42+epoch; shared across all variants/sizes',
                train_split='sample anchors from all contact episodes, both sensors; earliest-frame history padding',
                batch_unit='four-frame sensor clips; rae-1in1 processes frames independently',
                checkpoint_selector='minimum val MSE after each full epoch; test unused')
    if a.data_mode=='cached':
        config['cache_manifest_sha256']=hashlib.sha256((Path(a.cache_root)/'manifest.json').read_bytes()).hexdigest()
    if rank==0:
        (out/'config.json').write_text(json.dumps(config,indent=2))
        (out/'train_manifest.json').write_text(json.dumps([str(f) for f in ds.files],indent=2))
        print(json.dumps(dict(event='training_start',variant=a.variant,epochs=a.epochs,
                              sensor_anchors=len(ds),input_mode=spec['input_mode'],world_size=world)),flush=True)
    best=float('inf');best_epoch=None;step=resumed_step
    if (out/'best.json').exists():
        saved_best=json.loads((out/'best.json').read_text())
        best=saved_best['mse'];best_epoch=saved_best['epoch']
    completed_epoch=resumed_epoch
    stopped_early,_=early_stop_decision(dict(completed_epoch=resumed_epoch,best_epoch=best_epoch),a.patience)
    epochs=() if stopped_early else range(resumed_epoch+1,a.epochs+1)
    for epoch in epochs:
        sampler.set_epoch(epoch-1)
        step,train_mse,seen=train_epoch(model,loader,opt,device,accum=accum,micro_batch=a.micro_batch,
                                      samples_per_rank=len(sampler),start_step=step,epoch=epoch,
                                      rank=rank,world=world,log_path=out/'train.jsonl',precision=a.precision)
        completed_epoch=epoch
        stop_signal=torch.zeros(1,device=device,dtype=torch.uint8)
        if rank==0:
            metrics=evaluate(native,val_loader,device,a.precision)
            if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
                raise FloatingPointError('Nonfinite validation metrics')
            record=dict(epoch=epoch,step=step,train_mse=train_mse,sampled_clips=seen,**metrics)
            with (out/'val.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if metrics['mse']<best:
                best=metrics['mse'];best_epoch=epoch
                save_best(out,native,opt,epoch=epoch,step=step,metrics=metrics,spec=spec,config=config)
            save_checkpoint(out/'last.pt',native,opt,epoch=epoch,step=step,metrics=metrics,spec=spec,config=config)
            should_stop,stale=early_stop_decision(dict(completed_epoch=epoch,best_epoch=best_epoch),a.patience)
            if should_stop:stop_signal.fill_(1)
            status=dict(status='FinalEvaluation' if should_stop or epoch==a.epochs else 'Running',completed_epoch=epoch,
                        total_epochs=a.epochs,step=step,best_epoch=best_epoch,best_val_mse=best,
                        input_mode=spec['input_mode'],variant=a.variant,patience=a.patience,stale_epochs=stale)
            temp=out/'status.json.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(out/'status.json')
            print(json.dumps(dict(epoch_complete=record,best_epoch=best_epoch,best_val_mse=best)),flush=True)
        if world>1:dist.broadcast(stop_signal,src=0)
        if stop_signal.item():
            stopped_early=True
            break
    if rank==0:
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        native.load_state_dict(saved['model'],strict=True)
        test=holdout_class(a.eval_root,'test',a.interpolation,a.grid)
        test_metrics=evaluate(native,DataLoader(test,batch_size=1),device,a.precision)
        test.close()
        report=dict(best_epoch=best_epoch,best_val_mse=best,test=test_metrics)
        (out/'best_test.json').write_text(json.dumps(report,indent=2))
        status=dict(status='EarlyStopped' if stopped_early else 'Completed',completed_epoch=completed_epoch,total_epochs=a.epochs,
                    step=step,best_epoch=best_epoch,best_val_mse=best,input_mode=spec['input_mode'],
                    variant=a.variant,size=a.size,patience=a.patience,
                    stale_epochs=max(0,completed_epoch-best_epoch))
        temporary=out/'status.json.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(out/'status.json')
        print(json.dumps(dict(training_complete=status)),flush=True)
    if world>1:dist.barrier()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
