"""Real-weight GPU smoke: SSL backward, all-frame gradients, reload and policy tokens.

Run separately for each method/init to release all GPU memory between checks.
"""
import argparse, gc, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'encoder'))
import torch
from encoder.external import ExternalEncoder
from encoder.train_external import VisionAdaptation, reduce_replica_loss
from encoder.sparsh_training import SparshTraining
from policy.vla_common.tactile_encode import TactileEncoder
p=argparse.ArgumentParser(); p.add_argument('--method',required=True); p.add_argument('--init',required=True); args=p.parse_args()
torch.set_num_threads(4); torch.manual_seed(42)
from encoder.catalog import selection
mode, args.method, size, args.init = selection('univtac', args.method, None, args.init)
enc=ExternalEncoder(mode,args.method,size,pretrained=args.init in ('official','adapted'),with_decoder=True)
meta=dict(enc.metadata,input_mode='depth_deform_nonlinear_radial',initialization=args.init)
from encoder.linear_depth_deform import depth_deform_spec
meta['depth_deform_encoding']=depth_deform_spec(meta['input_mode'])
core=(SparshTraining(enc) if mode=='tactile' and args.method!='Anytouch2' else VisionAdaptation(enc)).cuda()
x=torch.rand(2,enc.num_frames,3,224,224,device='cuda',requires_grad=True) if enc.num_frames>1 else torch.rand(2,3,224,224,device='cuda',requires_grad=True)
opt=torch.optim.AdamW((p for p in core.parameters() if p.requires_grad),lr=1e-5)
before=next(enc.model.parameters()).detach().clone()
loss,metrics=core(x); loss,metrics=reduce_replica_loss(loss,metrics)
assert torch.isfinite(loss),loss
loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in core.parameters())
frame_grad=x.grad.abs().sum((0,2,3,4)) if x.ndim==5 else x.grad.abs().sum().reshape(1)
assert (frame_grad>0).all(),frame_grad
opt.step(); core.update_teacher()
assert not torch.equal(before,next(enc.model.parameters()))
core.eval()
with torch.no_grad():
    expected=enc.patch_tokens(x.detach()).cpu()
    val,_=core(x.detach()); assert torch.isfinite(val)
with tempfile.TemporaryDirectory() as d:
    ckpt=Path(d)/'encoder.pth'; torch.save(dict(meta,encoder=enc.state_dict()),ckpt)
    restored=ExternalEncoder(mode,args.method,size,metadata=meta,pretrained=False).cuda().eval()
    restored.load_state_dict(torch.load(ckpt,weights_only=True)['encoder'],strict=True)
    with torch.no_grad(): actual=restored.patch_tokens(x.detach()).cpu()
    torch.testing.assert_close(actual,expected)
    del restored; gc.collect(); torch.cuda.empty_cache()
    cfg=SimpleNamespace(tactile_encoder_path=str(ckpt),tactile_input_mode=meta['input_mode'],
         tactile_type='pool3x3',freeze_tactile_encoder=False,tactile_encoder_keys=lambda:['left'],tactile_encoder_config=None)
    adapter=TactileEncoder(cfg,64).cuda().eval()
    output=adapter.forward_flat({'left':x.detach()})
    assert output.shape[-1]==64 and torch.isfinite(output).all()
print(json.dumps(dict(method=args.method,init=args.init,loss=float(loss),frame_grad=frame_grad.tolist(),tokens=list(expected.shape),policy_tokens=list(output.shape),peak_gb=torch.cuda.max_memory_allocated()/2**30)),flush=True)
