"""Evaluate a matching RAE checkpoint on the frozen raw encoder val/test split."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from encoder.rae import RAE
from training.encoder_data import FrozenClips,CachedFrozenClips
from training.pretrain_rae import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--eval-root',default='data/encoder_eval_v1')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    spec=ck['encoder_spec']
    if spec['input_mode']!='depth_deform_radial' or spec['stride']!=2:
        raise ValueError('Checkpoint does not match the frozen depth_deform_radial stride-2 contract')
    interpolation=spec['representation']
    dataset_class=CachedFrozenClips if ck.get('config',{}).get('data_mode')=='cached' else FrozenClips
    ds=dataset_class(args.eval_root,args.split,interpolation['interpolation'],interpolation['coarse_grid'])
    model=RAE(variant=spec['variant'],size=spec.get('size','S'))
    precision=ck.get('config',{}).get('precision','amp_bf16')
    if precision=='bf16':model.to(torch.bfloat16)
    model.load_state_dict(ck['model'],strict=True);model.to(args.device)
    report=dict(checkpoint=args.checkpoint,step=ck['step'],split=args.split,clips=len(ds),
                sensor_frames=4*len(ds),encoder_spec=spec,
                **evaluate(model,DataLoader(ds,batch_size=1,num_workers=0),args.device,precision))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
