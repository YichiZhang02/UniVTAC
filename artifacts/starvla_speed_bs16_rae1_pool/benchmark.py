import argparse,json,time,statistics,traceback
from pathlib import Path
import torch
from training.models import StarVLA
from policy.starvla_pi import StarVLAPI
p=argparse.ArgumentParser();p.add_argument('--model',choices=['groot','pi'],required=True);p.add_argument('--tactile',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
cfg=json.loads(Path('training/config.json').read_text());cfg.update(tactile=a.tactile,encoder='rae',rae_variant='rae-1in1',tactile_frames=1,tactile_tokens='pool3x3',encoder_checkpoint=None)
record=dict(model=a.model,tactile=a.tactile,micro_batch=16,warmup_steps=1,measured_steps=3,
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,config=cfg,
            timing='resident synthetic batch; includes PIL/processor, forward, backward, clipping, AdamW; excludes data loading and DDP')
def save():Path(a.output).write_text(json.dumps(record,indent=2))
try:
 torch.manual_seed(42)
 torch.backends.cudnn.benchmark=False
 model=(StarVLAPI if a.model=='pi' else StarVLA)(cfg,8).cuda().train()
 vlm=list(model.native.qwen_vl_interface.parameters());ids={id(p) for p in vlm}
 other=[p for p in model.parameters() if id(p) not in ids]
 opt=torch.optim.AdamW([dict(params=vlm,lr=cfg['learning_rate']),dict(params=other,lr=cfg['action_learning_rate'])],weight_decay=1e-4)
 batch=dict(state=torch.randn(16,8,device='cuda'),action=torch.randn(16,50,8,device='cuda'),
            image=torch.rand(16,2,3,224,224,device='cuda'),
            tactile=torch.rand(16,2,1,3,224,224,device='cuda') if a.tactile else torch.empty(16,0,device='cuda'),
            lang=['Pick up the can and place it in the target area.']*16)
 def inspect_text(module,args,kwargs):
  h=args[0] if args else kwargs['hidden_states'];record['vlm_sequence_length']=h.shape[1]
 def inspect_head(module,args,kwargs):
  record['head_query_length']=args[0].shape[1];record['head_kv_length']=kwargs['encoder_hidden_states'].shape[1]
 model.native.qwen_vl_interface.model.model.language_model.layers[0].register_forward_pre_hook(inspect_text,with_kwargs=True)
 model.native.action_model.model.transformer_blocks[0].attn1.register_forward_pre_hook(inspect_head,with_kwargs=True)
 durations=[]
 for step in range(4):
  opt.zero_grad(set_to_none=True)
  if step==1:torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  print(json.dumps(dict(model=a.model,tactile=a.tactile,step=step,seconds=elapsed,loss=float(loss))),flush=True)
  if step>0:durations.append(elapsed)
 record.update(status='ok',step_seconds=durations,mean_seconds=statistics.mean(durations),
               median_seconds=statistics.median(durations),samples_per_second=16/statistics.mean(durations),
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
 save();print(json.dumps({k:v for k,v in record.items() if k!='config'}),flush=True)
except torch.OutOfMemoryError as e:
 record.update(status='oom',error=str(e));save();print('OOM',str(e),flush=True)
except Exception as e:
 record.update(status='error',error=str(e));save();traceback.print_exc();raise
