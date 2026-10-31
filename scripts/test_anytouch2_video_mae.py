"""Check tube masking, masked-only reconstruction and AnyTouch2 encoder export."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.Anytouch2.model import AnyTouchBackbone
from encoder.Anytouch2.training import VideoMAETraining, patchify_video, tube_mask, masked_pixel_loss


class VideoMAETest(unittest.TestCase):
    def test_patch_layout_and_masked_loss(self):
        x = torch.arange(1*4*3*4*4).reshape(1, 4, 3, 4, 4)
        target = patchify_video(x, 2, 2)
        self.assertEqual(tuple(target.shape), (1, 8, 24))
        torch.testing.assert_close(target[0, 0], x[0, :2, :, :2, :2].permute(0, 2, 3, 1).flatten())
        mask = torch.tensor([[False, True]])
        prediction = torch.tensor([[[999.], [2.]]], requires_grad=True)
        loss = masked_pixel_loss(prediction, torch.zeros_like(prediction), mask)
        self.assertEqual(loss.item(), 4.)
        loss.backward()
        self.assertEqual(prediction.grad[0, 0].item(), 0.)
        with self.assertRaisesRegex(ValueError, 'shapes must match'):
            masked_pixel_loss(prediction[0], torch.zeros_like(prediction), mask)

    def test_tube_mask_is_shared_across_time(self):
        ids, mask = tube_mask(3, 2, 14, 14, .75, 'cpu')
        self.assertEqual(tuple(ids.shape), (3, 98))
        torch.testing.assert_close(mask[:, :196], mask[:, 196:])
        self.assertTrue((mask.sum(1) == 294).all())
        self.assertFalse(mask.gather(1, ids).any())

    def test_half_mask_keeps_half_the_tokens(self):
        ids, mask = tube_mask(2, 2, 14, 14, .5, 'cpu')
        self.assertEqual(tuple(ids.shape), (2, 196))
        self.assertTrue((mask.sum(1) == 196).all())
        torch.testing.assert_close(mask[:, :196], mask[:, 196:])

    def test_removed_pixels_cannot_leak_and_encoder_gets_gradients(self):
        class Encoder(torch.nn.Module):
            hw = (32, 32)
            num_frames = 4
            embed_dim = 16

            def __init__(self):
                super().__init__()
                self.model = AnyTouchBackbone(dict(projection_dim=16, vision_config=dict(
                    image_size=32, patch_size=16, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=4)))

            def preprocess(self, x):
                return x

        torch.set_num_threads(2)
        encoder = Encoder()
        core = VideoMAETraining(encoder, .75, decoder_dim=32, decoder_layers=2, decoder_heads=4)
        images = torch.rand(2, 4, 3, 32, 32)
        ids, mask = tube_mask(2, 2, 2, 2, .75, 'cpu')
        changed = images.clone()
        pixel_mask = mask.reshape(2, 2, 2, 2).repeat_interleave(2, 1).repeat_interleave(16, 2).repeat_interleave(16, 3)
        changed.masked_fill_(pixel_mask[:, :, None], 999.)
        first, second = encoder.model(images, ids), encoder.model(changed, ids)
        self.assertEqual(first.shape[1], 8)  # six prefix tokens + two visible patches
        torch.testing.assert_close(first, second)
        loss, metrics = core(images)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad = encoder.model.touch_model.embeddings.patch_embedding.weight.grad
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
        self.assertTrue(core.decoder_pred.weight.grad.abs().sum() > 0)
        self.assertFalse(any('decoder' in key for key in encoder.state_dict()))
        self.assertFalse(hasattr(core, 'teacher'))
        self.assertEqual(metrics['mask_fraction'].item(), .75)
        if torch.cuda.is_available():
            core.zero_grad(set_to_none=True)
            core = core.cuda()
            cuda_loss, _ = core(images.cuda())
            self.assertTrue(torch.isfinite(cuda_loss))
            cuda_loss.backward()
            self.assertTrue(core.decoder_pred.weight.grad.abs().sum() > 0)


if __name__ == '__main__':
    unittest.main()
