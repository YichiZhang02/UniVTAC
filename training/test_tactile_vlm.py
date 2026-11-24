"""Synthetic checks for VTL packing, Qwen masks/RoPE and checkpointed gradients."""
import itertools
import unittest

import torch
from torch import nn

from training.tactile_vlm import TactileVLM, make_attention_mask


class TactileVLMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.embedding = nn.Embedding(32,32)
        # Same image block; two different language lengths with left padding.
        self.ids = torch.tensor([[0,1,10,11,11,12,5,6], [1,2,10,11,11,12,7,8]])
        self.valid = self.ids != 0
        self.positions = torch.stack([self.valid.long().cumsum(-1)-1]*3)
        self.inputs = dict(input_ids=self.ids,attention_mask=self.valid)

    def pack(self, wrapper='none',rope='1d',mode='causal',side=1):
        assembly=TactileVLM(dict(tactile_wrapper=wrapper,tactile_rope=rope,
                                vlm_attention_mask=mode),32)
        tactile=torch.randn(2,2*side*side,32,requires_grad=True)
        packed=assembly.pack(self.inputs,self.embedding(self.ids),self.positions,
                             tactile,(2,1,side,side),10,12,self.embedding)
        return assembly,packed,tactile

    def test_all_wrappers_positions_and_masks(self):
        for wrapper,rope,mode,side in itertools.product(
                ('none','vision','tactile'),('none','1d','3d'),
                ('causal','modality','bidirectional'),(1,3,7,14)):
            with self.subTest(wrapper=wrapper,rope=rope,mode=mode,side=side):
                assembly,packed,tactile=self.pack(wrapper,rope,mode,side)
                vl,t=packed.split(packed.embeddings)
                torch.testing.assert_close(vl,self.embedding(self.ids))
                torch.testing.assert_close(t,tactile)
                self.assertEqual(packed.tactile_body.sum(1).tolist(),[2*side*side]*2)
                expected=2*side*side+(4 if wrapper != 'none' else 0)
                self.assertEqual(packed.tactile_span.sum(1).tolist(),[expected]*2)
                self.assertEqual(packed.positions.shape,(3,2,8+expected))
                mask=make_attention_mask(packed.valid,packed.modalities,mode,torch.float32)[:,0]==0
                self.assertFalse(mask[0,:,0].any())
                tq=int(torch.where(packed.tactile_body[1])[0][0]);lq=mask.shape[-1]-1
                self.assertEqual(bool(mask[1,tq,lq]),mode=='bidirectional')
                self.assertTrue(mask[1,lq,tq])
                if rope=='none':
                    self.assertTrue((packed.positions[:,packed.tactile_span]==0).all())
                elif rope=='1d':
                    tp=packed.positions[:,1,packed.tactile_span[1]]
                    self.assertTrue(torch.equal(tp[0],tp[1]) and torch.equal(tp[1],tp[2]))
                elif wrapper=='none':
                    tp=packed.positions[:,1,packed.tactile_body[1]]
                    self.assertTrue(torch.equal(tp[:,:side*side],tp[:,side*side:]))

    def test_real_hybrid_text_layers(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        config=Qwen3_5TextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            layer_types=['linear_attention']*3+['full_attention'],
            linear_num_key_heads=2,linear_num_value_heads=2,
            linear_key_head_dim=8,linear_value_head_dim=8,
            rope_parameters={'rope_type':'default','rope_theta':10000.,
                             'partial_rotary_factor':.5,'mrope_section':[2,1,1]})
        config._attn_implementation='sdpa'
        text=Qwen3_5TextModel(config).eval()
        # This environment has CUDA-only optional kernels installed. Use the
        # upstream mathematical fallback for this CPU test, for both paths.
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
        for layer in text.layers:
            if layer.layer_type == 'linear_attention':
                layer.linear_attn.norm=hf.Qwen3_5RMSNormGated(8)
                layer.linear_attn.causal_conv1d_fn=None
                layer.linear_attn.chunk_gated_delta_rule=hf.torch_chunk_gated_delta_rule
        # With causal masking the custom runner matches native HF execution.
        assembly,packed,_=self.pack()
        actual=assembly.run_text(text,packed)[-1]
        expected=text(inputs_embeds=packed.embeddings,position_ids=packed.positions,
                      attention_mask=packed.valid,use_cache=False).last_hidden_state
        torch.testing.assert_close(actual[packed.valid],expected[packed.valid])
        # Check all 27 VLM options on real hybrid layers, including gradients.
        for wrapper,rope,mode in itertools.product(('none','vision','tactile'),
                ('none','1d','3d'),('causal','modality','bidirectional')):
            assembly,packed,tactile=self.pack(wrapper,rope,mode,side=3)
            out=assembly.run_text(text,packed)
            self.assertEqual(len(out),4)
            loss=out[-1].square().mean();self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(tactile.grad.abs().sum().item(),0.)
            if assembly.delimiters is not None:
                self.assertGreater(assembly.delimiters.weight.grad.abs().sum().item(),0.)
            text.zero_grad(set_to_none=True)
        # Qwen's own checkpointing layer wrapper must preserve all gradients.
        text.train();text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        assembly,packed,tactile=self.pack('tactile','3d','modality')
        assembly.run_text(text,packed)[-1].square().mean().backward()
        self.assertGreater(tactile.grad.abs().sum().item(),0.)


    def test_real_vision_and_no_tactile_mask_path(self):
        from types import SimpleNamespace
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
        config=Qwen3_5Config(
            text_config=dict(vocab_size=32,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
                layer_types=['full_attention'],rope_parameters={'rope_type':'default',
                    'rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[2,1,1]}),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
                out_hidden_size=32,patch_size=16,temporal_patch_size=2,
                spatial_merge_size=2,num_position_embeddings=16),
            vision_start_token_id=10,vision_end_token_id=12,image_token_id=11,video_token_id=13)
        config._attn_implementation='sdpa'
        model=hf.Qwen3_5ForConditionalGeneration(config).eval()
        ids=torch.tensor([[1,10,11,12,10,11,12,5]])
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                    pixel_values=torch.randn(8,3*2*16*16),
                    image_grid_thw=torch.tensor([[1,2,2],[1,2,2]]),
                    mm_token_type_ids=(ids==11).long())
        vlm=SimpleNamespace(model=model,build_qwenvl_inputs=lambda *args:inputs)
        for mode in ('causal','modality','bidirectional'):
            assembly=TactileVLM(dict(vlm_attention_mask=mode),32)
            layers,packed=assembly(vlm,None,None)
            self.assertEqual(layers[-1].shape,(1,8,32))
            self.assertFalse(packed.tactile_span.any())
            self.assertEqual(packed.split(layers[-1])[1].shape,(1,0,32))
            tactile=torch.randn(1,2,32,requires_grad=True)
            layers,packed=assembly(vlm,None,None,tactile,(2,1,1,1))
            self.assertEqual(layers[-1].shape,(1,10,32))
            layers[-1].square().mean().backward()
            self.assertGreater(tactile.grad.abs().sum().item(),0.)
            model.zero_grad(set_to_none=True)


if __name__=='__main__':
    unittest.main()
