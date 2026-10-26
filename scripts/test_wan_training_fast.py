"""Verify full-clip Wan's temporal boundary, gradients and checkpoint contract."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import AutoencoderKLWan
from encoder.wan_training_fast import enable_wan_training_fast


def tiny_model():
    torch.manual_seed(42)
    return AutoencoderKLWan(base_dim=8, decoder_base_dim=8, z_dim=4,
        dim_mult=[1, 2, 2, 2], num_res_blocks=1,
        temperal_downsample=[False, True, True], patch_size=2,
        in_channels=12, out_channels=12, is_residual=True)


class WanTrainingFastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_five_frame_values_and_gradients(self):
        reference = tiny_model().train()
        fast = enable_wan_training_fast(copy.deepcopy(reference), compile_encoder=False, channels_last=False)
        x = torch.randn(1, 3, 5, 32, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        expected = reference.encode(x).latent_dist.mean
        actual = fast.encode(y).latent_dist.mean
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream)
        torch.testing.assert_close(y.grad, x.grad, rtol=3e-4, atol=3e-5)
        for (name, old), (new_name, new) in zip(reference.named_parameters(), fast.named_parameters()):
            self.assertEqual(name, new_name)
            if old.grad is None:
                self.assertIsNone(new.grad)
            else:
                torch.testing.assert_close(new.grad, old.grad, rtol=3e-4, atol=3e-5, msg=name)

    def test_inference_and_nonfive_frame_fallback(self):
        reference = tiny_model()
        fast = enable_wan_training_fast(copy.deepcopy(reference), compile_encoder=False, channels_last=False)
        for training, frames in ((False, 5), (True, 9), (True, 1)):
            reference.train(training)
            fast.train(training)
            x = torch.randn(1, 3, frames, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(fast.encode(x).latent_dist.mean,
                                           reference.encode(x).latent_dist.mean, rtol=0, atol=0)

    def test_state_dict_contract(self):
        reference = tiny_model()
        before = {key: (value.shape, value.dtype) for key, value in reference.state_dict().items()}
        fast = copy.deepcopy(reference)
        parameter_ids = [id(p) for p in fast.parameters()]
        trainable = [p.requires_grad for p in fast.parameters()]
        enable_wan_training_fast(fast)
        self.assertEqual(parameter_ids, [id(p) for p in fast.parameters()])
        self.assertEqual(trainable, [p.requires_grad for p in fast.parameters()])
        after = {key: (value.shape, value.dtype) for key, value in fast.state_dict().items()}
        self.assertEqual(before, after)
        fast.load_state_dict(reference.state_dict(), strict=True)
        reference.load_state_dict(fast.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
