"""Physical representation, recorded/live parity, RAE topology and loading contracts."""
import gc
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from encoder.rae import RAE
from encoder.tactile import convert,dense_displacement,from_observation,representation_spec
from training.models import Tactile


def main():
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):
        xx,yy=np.meshgrid(np.linspace(20,300,7),np.linspace(20,220,9))
        fixed=np.c_[xx.ravel(),yy.ravel()]
        disp=np.c_[.02*fixed[:,0]+.03*fixed[:,1]+2,-.01*fixed[:,0]+.04*fixed[:,1]-1]
        markers=np.stack((fixed,fixed+disp)).astype(np.float32)
        depth=np.full((240,320),28.,np.float32)
        for method in ('linear','tps_bilinear','tps'):
            field=dense_displacement(markers,depth.shape,size=40,method=method)
            qx,qy=np.meshgrid((np.arange(40)+.5)*320/40,(np.arange(40)+.5)*240/40)
            expected=np.stack((.02*qx+.03*qy+2,-.01*qx+.04*qy-1),-1)
            np.testing.assert_allclose(field,expected,atol=.003)
            represented=convert(depth,markers,size=40,method=method)
            torch.testing.assert_close(represented[0],torch.full((40,40),.4))
        zero=convert(depth,np.stack((fixed,fixed)).astype(np.float32))
        torch.testing.assert_close(zero[1:],torch.full_like(zero[1:],.5))
        huge=np.stack((fixed,fixed+np.array([60.,80.]))).astype(np.float32)
        encoded=convert(depth,huge)
        torch.testing.assert_close(encoded[1],torch.full_like(encoded[1],.8))
        torch.testing.assert_close(encoded[2],torch.full_like(encoded[2],.9))
        # Duplicate padded markers must not change the result.
        duplicate=np.concatenate((markers,markers[:,:1]),axis=1)
        torch.testing.assert_close(convert(depth,markers),convert(depth,duplicate))
        np.testing.assert_allclose(from_observation({'depth':torch.from_numpy(depth),'marker':torch.from_numpy(markers)}),convert(depth,markers))
        ref=Path('/mnt/yichi_data/UniVTAC1/encoder')
        if ref.exists():
            pkg=types.ModuleType('univtac1_reference');pkg.__path__=[str(ref)]
            sys.modules[pkg.__name__]=pkg
            reference=importlib.import_module('univtac1_reference.radial_depth_deform')
            # Non-affine real markers, including a duplicate in contact data.
            path=next(Path('/mnt/yichi_data/UniVTAC_downloads/data/contact').glob('*/hdf5/*.hdf5'))
            with h5py.File(path) as f:
                g=f['tactile/left_tactile'];d=g['depth'][100];m=g['marker'][100]
            expected=reference.convert_radial(d,m)
            np.testing.assert_allclose(convert(d,m,method='tps').numpy(),expected,atol=2e-6)
        print('PASS: radial/depth semantics, affine motion, saturation direction, duplicate controls, live parity and UniVTAC1 dense TPS parity',flush=True)
        for variant,n,t in [('rae-1in1',196,1),('rae-4in1',196,4)]:
            torch.manual_seed(42)
            model=RAE(variant=variant)
            x=torch.rand(1,4,3,224,224)
            y=model(x)
            assert y.shape==x.shape and torch.isfinite(y).all()
            (y-x).square().mean().backward()
            assert model.encoder.patch.weight.grad is not None
            if variant=='rae-4in1':assert model.encoder.compress.weight.grad is not None
            with torch.no_grad():
                single=x[:,:t]
                for mode,count in [('full',n),('pool3x3',9),('cls',1)]:
                    assert model.encoder.tokens(single,mode).shape==(1,count,384)
                if variant=='rae-1in1':
                    model.eval()
                    torch.testing.assert_close(model(x)[:,2:3],model(x[:,2:3]),atol=2e-6,rtol=2e-5)
                else:
                    # Temporal frame identities exist as ordinary additive parameters.
                    assert model.encoder.temporal_pe.shape==(1,4,1,384)
            assert not any('rope' in name.lower() for name,_ in model.named_modules())
            with tempfile.TemporaryDirectory() as tmp:
                ck=Path(tmp)/'encoder.pt'
                spec=dict(variant=variant,input_mode='depth_deform_radial',stride=2,representation=representation_spec())
                torch.save(dict(encoder=model.encoder.state_dict(),encoder_spec=spec),ck)
                cfg=dict(encoder='rae',tactile_tokens='full',tactile_frames=t,tactile_stride=2,
                         rae_variant=variant,encoder_checkpoint=str(ck),tactile_input_mode='depth_deform_radial')
                adapter=Tactile(cfg)
                with torch.no_grad():
                    tokens=adapter(x[:,:t,None].transpose(1,2).expand(-1,2,-1,-1,-1,-1))
                    assert tokens.shape==(1,n*2,384)
                del adapter
            print(f'PASS: {variant}, recon/backward, tokens, independent single-frame semantics, checkpoint/policy loading',flush=True)
            del model,x,y;gc.collect()


if __name__=='__main__':main()
