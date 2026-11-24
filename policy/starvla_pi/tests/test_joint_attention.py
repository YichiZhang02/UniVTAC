"""Run with the training Python: python -m unittest discover -s policy/starvla_pi/tests."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class JointAttentionTest(unittest.TestCase):
    def test_joint_keys_mask_and_gradients(self):
        torch.manual_seed(7)
        model = DiT(num_attention_heads=2, attention_head_dim=16,
                    cross_attention_dim=32, num_layers=2, output_dim=32,
                    dropout=0., final_dropout=False, positional_embeddings=None,
                    joint_action_attention=True).eval()
        action = torch.randn(2, 3, 32, requires_grad=True)
        conditions = [torch.randn(2, 4, 32, requires_grad=True) for _ in range(2)]
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        timestep = torch.tensor([100, 500])
        observed = []
        def inspect(module, args, kwargs):
            observed.append((args[0].shape, kwargs['encoder_hidden_states'].shape,
                             kwargs['attention_mask'].clone()))
        hooks = [b.attn1.register_forward_pre_hook(inspect, with_kwargs=True)
                 for b in model.transformer_blocks]
        def run(a, c):
            return model(a, c, timestep, encoder_attention_mask=mask, return_pre_output=True)
        output = run(action, conditions)
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(observed), 2)
        for query_shape, kv_shape, full_mask in observed:
            self.assertEqual(query_shape, (2, 3, 32))
            self.assertEqual(kv_shape, (2, 7, 32))
            self.assertTrue(torch.equal(full_mask[:, :4], mask))
            self.assertTrue(full_mask[:, 4:].all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        for cond in conditions:
            self.assertGreater(cond.grad[mask].abs().sum().item(), 0.)
            self.assertEqual(cond.grad[~mask].abs().sum().item(), 0.)
        altered = [torch.where(mask[..., None], c.detach(), torch.randn_like(c)*100)
                   for c in conditions]
        torch.testing.assert_close(output, run(action, altered))
        changed_action = action.detach().clone()
        changed_action[:, -1] += torch.randn(2, 32)*5
        self.assertFalse(torch.allclose(output[:, 0], run(changed_action, conditions)[:, 0]))
        self.assertEqual(run(action, conditions).shape, (2, 3, 32))


class GR00TQueryTest(unittest.TestCase):
    def test_ca_only_actions_sa_keeps_state(self):
        model = DiT(num_attention_heads=2, attention_head_dim=16,
                    cross_attention_dim=32, num_layers=4, output_dim=32,
                    dropout=0., final_dropout=False, positional_embeddings=None,
                    interleave_self_attention=True, action_query_count=3).eval()
        shapes=[]
        def inspect(module,args,kwargs):
            kv=kwargs['encoder_hidden_states']
            shapes.append((args[0].shape[1], None if kv is None else kv.shape[1]))
        handles=[b.attn1.register_forward_pre_hook(inspect,with_kwargs=True)
                 for b in model.transformer_blocks]
        output=model(torch.randn(2,4,32),torch.randn(2,5,32),torch.tensor([1,2]),
                     encoder_attention_mask=torch.ones(2,5,dtype=torch.bool),return_pre_output=True)
        self.assertEqual(shapes,[(3,5),(4,None),(3,5),(4,None)])
        self.assertEqual(output.shape,(2,4,32))
        for h in handles:h.remove()


if __name__ == '__main__':
    unittest.main()
