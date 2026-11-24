"""CNN initialization, frozen-encoder training and strict reload consistency."""
import gc
import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from encoder.token_pooling import Learned3x3
from training.models import Tactile


class LearnedPoolingTest(unittest.TestCase):
    def test_exact_pool_initialization_and_two_step_gradients(self):
        torch.manual_seed(42)
        for channels,side in ((384,14),(768,14),(512,7)):
            layer=Learned3x3(channels)
            x=torch.randn(2,channels,side,side,requires_grad=True)
            torch.testing.assert_close(layer(x),F.adaptive_avg_pool2d(x,3),rtol=0,atol=0)
            target=torch.randn(2,channels,3,3)
            opt=torch.optim.SGD(layer.parameters(),lr=.1)
            (layer(x)-target).square().mean().backward()
            self.assertGreater(layer.mix.weight.grad.abs().sum().item(),0)
            self.assertEqual(layer.spatial.weight.grad.abs().sum().item(),0)
            opt.step();opt.zero_grad(set_to_none=True)
            (layer(x)-target).square().mean().backward()
            self.assertGreater(layer.spatial.weight.grad.abs().sum().item(),0)
            self.assertGreater(x.grad.abs().sum().item(),0)

    def test_real_rae_variants_freezing_and_reload(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        # On CPU the S/1in1 cases suffice for gradient/reload logic. The larger
        # architectural matrix is exercised on CUDA when available.
        variants=[('S','rae-1in1'),('S','rae-4in1'),('B','rae-1in1'),('B','rae-4in1')]
        if device=='cpu':variants=variants[:1]
        for size,variant in variants:
            frames=1 if variant=='rae-1in1' else 4
            for frozen in (False,True):
                with self.subTest(size=size,variant=variant,frozen=frozen):
                    cfg=dict(encoder='rae',rae_size=size,rae_variant=variant,tactile_frames=frames,
                             tactile_stride=2,tactile_tokens='learned3x3',encoder_checkpoint=None,
                             freeze_tactile_encoder=frozen)
                    tactile=Tactile(cfg).to(device).train()
                    project=nn.Linear(tactile.dim,8).to(device)
                    x=torch.randn(1,2,frames,3,224,224,device=device)
                    self.assertEqual(tactile.encoder.training,not frozen)
                    self.assertTrue(tactile.downsample.training)
                    self.assertTrue(all(p.requires_grad != frozen for p in tactile.encoder.parameters()))
                    self.assertTrue(all(p.requires_grad for p in tactile.downsample.parameters()))
                    opt=torch.optim.SGD(list(tactile.parameters())+list(project.parameters()),lr=.01)
                    tokens=tactile(x)
                    self.assertEqual(tokens.shape,(1,18,tactile.dim))
                    self.assertEqual(tactile.token_layout(x),(2,1,3,3))
                    project(tokens).square().mean().backward()
                    self.assertGreater(project.weight.grad.abs().sum().item(),0)
                    self.assertGreater(tactile.downsample.mix.weight.grad.abs().sum().item(),0)
                    if frozen:
                        self.assertTrue(all(p.grad is None for p in tactile.encoder.parameters()))
                    else:
                        self.assertGreater(tactile.encoder.patch.weight.grad.abs().sum().item(),0)
                        if frames==4:
                            self.assertGreater(tactile.encoder.compress.weight.grad.abs().sum().item(),0)
                    opt.step();opt.zero_grad(set_to_none=True)
                    # One trained case validates strict checkpoint reload of both
                    # RAE and the nonzero CNN, without a VLM download or simulator.
                    if size=='S' and frames==1 and frozen:
                        tactile.eval()
                        with torch.no_grad():expected=tactile(x)
                        payload=io.BytesIO();torch.save(tactile.state_dict(),payload);payload.seek(0)
                        restored=Tactile(cfg).to(device).eval()
                        restored.load_state_dict(torch.load(payload,map_location=device,weights_only=True),strict=True)
                        with torch.no_grad():actual=restored(x)
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                        del restored,actual,expected,payload
                    del tactile,project,opt,tokens,x
                    gc.collect()
                    if device=='cuda':torch.cuda.empty_cache()
        # Freeze also applies to the existing pool route, without CNN parameters.
        cfg=dict(encoder='rae',rae_size='S',rae_variant='rae-1in1',tactile_frames=1,
                 tactile_tokens='pool3x3',encoder_checkpoint=None,freeze_tactile_encoder=True)
        tactile=Tactile(cfg).train()
        self.assertFalse(tactile.encoder.training)
        self.assertIsNone(tactile.downsample)
        self.assertTrue(all(not p.requires_grad for p in tactile.encoder.parameters()))


if __name__=='__main__':unittest.main()
