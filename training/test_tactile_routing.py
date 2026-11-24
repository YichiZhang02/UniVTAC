"""Ensure VLM fusion passes each complete VTL sequence to the action head."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'policy/starvla_pi/vendor'))
from policy.starvla_pi import StarVLAPI
from training.models import StarVLA
from training.tactile_vlm import TactileVLM


class FakeTactile(nn.Module):
    def forward(self,x):return x
    def token_layout(self,x):return (2,1,1,1)


class FakeFusion(nn.Module):
    def __init__(self):
        super().__init__()
        assembly=TactileVLM({'tactile_wrapper':'tactile'},32)
        ids=torch.tensor([[1,10,11,12,5]])
        self.packed=assembly.pack(dict(input_ids=ids,attention_mask=torch.ones_like(ids)),
            torch.zeros(1,5,32),torch.arange(5)[None,None].expand(3,1,5),
            torch.zeros(1,2,32),(2,1,1,1),10,12,nn.Embedding(32,32))
        self.layers=[torch.full_like(self.packed.embeddings,float(i)) for i in (1,2,3)]
    def forward(self,*args):return self.layers,self.packed


class RoutingTest(unittest.TestCase):
    def test_layer_selection(self):
        batch=dict(image=torch.zeros(1,2,3,4,4),tactile=torch.zeros(1,2,32),
                   lang=['test'],state=torch.zeros(1,8))
        for cls in (StarVLA,StarVLAPI):
            model=cls.__new__(cls);nn.Module.__init__(model)
            model.fusion_mode='vlm';model.tactile=FakeTactile();model.project=nn.Identity()
            model.vlm_tactile=FakeFusion()
            model.native=SimpleNamespace(qwen_vl_interface=None,
                add_discretized_state_to_instruction=lambda lang,state:lang,
                _project_vl_hidden_for_action=lambda layers:layers)
            if cls is StarVLAPI:
                model.inject_tactile_vlm=True
                conditions,mask=model.encode(batch)
                self.assertEqual(len(conditions),3)
                for i,h in enumerate(conditions,1):
                    self.assertTrue((h==i).all())
                    self.assertEqual(h.shape,(1,11,32))
            else:
                h,mask=model.encode(batch)
                self.assertEqual(h.shape,(1,11,32))
                self.assertTrue((h==3).all())
            self.assertEqual(mask.shape,(1,11))
            self.assertTrue(mask.all())


if __name__=='__main__':unittest.main()
