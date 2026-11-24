"""Forward/backward contract checks without accessing any legacy data."""
from training.tactile_vlm import add_tactile_arguments, apply_tactile_arguments
import argparse
import json
import torch
from training.models import ACT,StarVLA
from policy.starvla_pi import StarVLAPI
from policy.starvla_groot_v1 import StarVLAGrootV1
from encoder.rae import Encoder,RAE

def main():
    p=argparse.ArgumentParser();add_tactile_arguments(p);p.add_argument('--model',choices=['act','starvla','starvla_pi','starvla_groot_v1','rae'],required=True);p.add_argument('--tactile',action='store_true');p.add_argument('--encoder',choices=['univtac','rae'],default='rae');p.add_argument('--tokens',choices=['full','pool3x3','learned3x3','cls'],default='pool3x3');p.add_argument('--rae-variant',choices=['rae-1in1','rae-4in1'],default='rae-4in1');p.add_argument('--rae-size',choices=['S','B'],default='S');a=p.parse_args()
    torch.manual_seed(42)
    if a.model=='rae':
        model=RAE(variant=a.rae_variant,size=a.rae_size).cuda()
        x=torch.rand(1,4,3,224,224,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=model(x);loss=(y-x).square().mean()
        loss.backward()
        assert model.encoder.patch.weight.grad is not None
        with torch.no_grad():
            for mode,n in [('full',196*model.encoder.latent_frames),('pool3x3',9*model.encoder.latent_frames),('cls',1)]:
                assert model.encoder.tokens(x[:,:model.frames],mode).shape==(1,n,model.encoder.embed_dim)
        print('RAE reconstruction/backward and token modes passed',flush=True);return
    cfg=json.load(open('training/config.json'));apply_tactile_arguments(a,cfg);cfg['tactile']=a.tactile;cfg['encoder']=a.encoder;cfg['tactile_tokens']=a.tokens
    cfg['rae_variant']=a.rae_variant;cfg['rae_size']=a.rae_size
    if a.encoder=='rae':cfg['tactile_frames']=1 if a.rae_variant=='rae-1in1' else 4
    model=({'act': ACT, 'starvla': StarVLA, 'starvla_pi': StarVLAPI, 'starvla_groot_v1': StarVLAGrootV1}[a.model])(cfg,8).cuda()
    b=dict(state=torch.randn(1,8,device='cuda'),action=torch.randn(1,50,8,device='cuda'),
           image=torch.rand(1,2,3,224,224,device='cuda'),tactile=torch.rand(1,2,cfg['tactile_frames'],3,224,224,device='cuda') if a.tactile else torch.empty(1,0,device='cuda'),
           is_pad=torch.zeros(1,50,dtype=torch.bool,device='cuda'),lang=['Lift the can.'])
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(b)
    loss.backward();assert torch.isfinite(loss)
    print(a.model,'forward/backward passed',float(loss),flush=True)
if __name__=='__main__':main()
