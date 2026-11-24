"""Real Episodes/DataLoader timing; use the current working repository."""
import argparse,json,time,statistics,traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader,DistributedSampler,get_worker_info
from training.models import StarVLA
from policy.starvla_pi import StarVLAPI
from training.data import Episodes
from training.encoder_data import initialize_worker

class ProfiledEpisodes(Episodes):
    def __getitem__(self,index):
        self.convert_seconds=0.
        start=time.perf_counter()
        out=super().__getitem__(index)
        out['_item_seconds']=time.perf_counter()-start
        out['_convert_seconds']=self.convert_seconds
        return out

def profile_worker(worker_id):
    initialize_worker(worker_id)
    import encoder.tactile as tactile_module
    ds=get_worker_info().dataset
    original=tactile_module.convert
    def measured(*args,**kwargs):
        start=time.perf_counter()
        out=original(*args,**kwargs)
        ds.convert_seconds+=time.perf_counter()-start
        return out
    tactile_module.convert=measured

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['groot','pi'],required=True);p.add_argument('--tactile',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    root=Path('artifacts/starvla_e2e_bs16_rae1_pool_no_checkpoint')
    cfg=json.loads(Path('training/config.json').read_text())
    cfg.update(tactile=a.tactile,encoder='rae',rae_variant='rae-1in1',tactile_frames=1,
               tactile_tokens='pool3x3',encoder_checkpoint=None)
    record=dict(model=a.model,tactile=a.tactile,micro_batch=16,workers=4,prefetch_factor=2,
                warmup_steps=2,measured_steps=10,gpu=torch.cuda.get_device_name(),config=cfg,
                method='Real HDF5 Episodes; next(DataLoader) + H2D + forward + backward + clipping + AdamW; steady-state prefetch; no DDP communication')
    def save():Path(a.output).write_text(json.dumps(record,indent=2))
    try:
        torch.manual_seed(42);torch.backends.cudnn.benchmark=False
        ds=ProfiledEpisodes(cfg['data_root'],'instructions',cfg['chunk'],a.tactile,1,cfg['tactile_stride'],
                            cfg['tactile_input_mode'],cfg['tactile_interpolation'],cfg['tactile_interpolation_grid'])
        ds.stats=json.loads((root/'stats.json').read_text())
        sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=42);sampler.set_epoch(0)
        loader=DataLoader(ds,batch_size=16,sampler=sampler,num_workers=4,pin_memory=True,
                          drop_last=True,worker_init_fn=profile_worker,prefetch_factor=2)
        model=(StarVLAPI if a.model=='pi' else StarVLA)(cfg,8).cuda().train()
        assert cfg['flow_repeats'] == 8
        assert not model.native.qwen_vl_interface.model.is_gradient_checkpointing
        record['gradient_checkpointing_active'] = model.native.qwen_vl_interface.model.is_gradient_checkpointing
        head_batches = []
        hook = model.native.action_model.model.transformer_blocks[0].attn1.register_forward_pre_hook(
            lambda module,args,kwargs: head_batches.append(args[0].shape[0]), with_kwargs=True)
        vlm=list(model.native.qwen_vl_interface.parameters());ids={id(p) for p in vlm}
        opt=torch.optim.AdamW([dict(params=vlm,lr=cfg['learning_rate']),
                dict(params=[p for p in model.parameters() if id(p) not in ids],lr=cfg['action_learning_rate'])],weight_decay=1e-4)
        iterator=iter(loader);rows=[]
        for step in range(12):
            opt.zero_grad(set_to_none=True)
            if step==2:torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            batch=next(iterator)
            wait=time.perf_counter()-start
            item_cpu=float(batch.pop('_item_seconds').sum())
            convert_cpu=float(batch.pop('_convert_seconds').sum())
            batch={k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            row=dict(step=step,total_seconds=elapsed,data_wait_seconds=wait,
                     transfer_compute_update_seconds=elapsed-wait,
                     worker_item_cpu_seconds=item_cpu,worker_conversion_cpu_seconds=convert_cpu,loss=float(loss))
            print(json.dumps(row),flush=True)
            if step>=2:rows.append(row)
        total=[r['total_seconds'] for r in rows];waits=[r['data_wait_seconds'] for r in rows]
        record.update(status='ok',episodes=len(ds.files),frames=len(ds),steps=rows,
                      mean_seconds=statistics.mean(total),median_seconds=statistics.median(total),
                      mean_data_wait_seconds=statistics.mean(waits),median_data_wait_seconds=statistics.median(waits),
                      mean_compute_seconds=statistics.mean(r['transfer_compute_update_seconds'] for r in rows),
                      mean_conversion_cpu_ms_per_sample=sum(r['worker_conversion_cpu_seconds'] for r in rows)/160*1000,
                      mean_loading_preprocessing_cpu_ms_per_sample=sum(r['worker_item_cpu_seconds'] for r in rows)/160*1000,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        record['training_head_batches'] = sorted(set(head_batches))
        assert record['training_head_batches'] == [128]
        model.zero_grad(set_to_none=True);model.eval();head_batches.clear()
        from unittest.mock import patch
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16), patch('torch.randn', wraps=torch.randn) as noise:
            predicted = model.predict_action(batch)
        assert predicted.shape == (16,50,8) and torch.isfinite(predicted).all()
        assert noise.call_count == 1, noise.call_count
        assert set(head_batches) == {16}
        record.update(inference_noise_draws=noise.call_count,inference_head_batches=sorted(set(head_batches)),
                      inference_output_shape=list(predicted.shape))
        hook.remove()
        save();print(json.dumps({k:v for k,v in record.items() if k not in ('config','steps')}),flush=True)
        del iterator,loader
    except torch.OutOfMemoryError as e:
        record.update(status='oom',error=str(e));save();print('OOM',str(e),flush=True)
    except Exception as e:
        record.update(status='error',error=str(e));save();traceback.print_exc();raise

if __name__=='__main__':main()
