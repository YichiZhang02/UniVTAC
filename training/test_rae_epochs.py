"""Epoch-tail accumulation and complete best-model reload checks."""
import copy
import gc
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from encoder.rae import RAE
from encoder.tactile import representation_spec
from training.pretrain_rae import train_epoch
from training.rae_checkpoint import save_best


def main():
    torch.set_num_threads(2);torch.manual_seed(7)
    x=torch.randn(11,3)
    actual=nn.Linear(3,3);expected=copy.deepcopy(actual)
    opt=torch.optim.AdamW(actual.parameters(),lr=1e-3)
    ref=torch.optim.AdamW(expected.parameters(),lr=1e-3)
    for chunk in (x[:6],x[6:]):
        ref.zero_grad();(expected(chunk)-chunk).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(),1.);ref.step()
    step,loss,count=train_epoch(actual,DataLoader(x,batch_size=2),opt,'cpu',accum=3,
                               micro_batch=2,samples_per_rank=11)
    assert step==2 and count==11
    for a,b in zip(actual.parameters(),expected.parameters()):torch.testing.assert_close(a,b)
    print('PASS: full epoch consumes all 11 samples, correct partial final accumulation group',flush=True)
    for size,variant in [('S','rae-1in1'),('S','rae-4in1'),('B','rae-1in1'),('B','rae-4in1')]:
        model=RAE(variant=variant,size=size).eval();optimizer=torch.optim.AdamW(model.parameters())
        spec=dict(variant=variant,size=size,input_mode='depth_deform_radial',stride=2,representation=representation_spec())
        with tempfile.TemporaryDirectory() as tmp:
            save_best(tmp,model,optimizer,epoch=3,step=6,metrics=dict(mse=.1),spec=spec,config={})
            full=torch.load(Path(tmp)/'best_model.pt',map_location='cpu',weights_only=False)
            payload=torch.load(Path(tmp)/'best.pt',map_location='cpu',weights_only=False)
            reconstructed=RAE(**payload['architecture']['kwargs']).eval()
            reconstructed.load_state_dict(payload['model'],strict=True)
            z=torch.rand(1,4,3,224,224)
            with torch.no_grad():
                torch.testing.assert_close(full(z),model(z),rtol=0,atol=0)
                torch.testing.assert_close(reconstructed(z),model(z),rtol=0,atol=0)
            assert payload['epoch']==3 and payload['encoder_spec']==spec
            assert (Path(tmp)/'model_source/encoder/rae.py').exists()
        print(f'PASS: ViT-{size} {variant} best epoch saves/reloads full encoder+decoder and architecture-based model',flush=True)
        del model,optimizer,full,payload,reconstructed,z
        gc.collect()


if __name__=='__main__':main()
