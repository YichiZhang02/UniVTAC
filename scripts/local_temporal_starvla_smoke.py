"""Two optimizer steps and validation for a pinned tactile StarVLA encoder."""
import argparse
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import torch
    import yaml
    import policy.train_unified as trainer
    from policy.vla_common.tactile_encode import TactileEncoder
    from encoder.catalog import EXTERNAL_METHODS

    external = args.method in EXTERNAL_METHODS
    tokens_per_sensor = ({'Sparsh_mae': 9, 'Sparsh_DINOv2': 10, 'Sparsh_IJEPA': 10,
                          'Sparsh_VJEPA': 19, 'Anytouch2': 19}[args.method] if external else 19)

    saved = yaml.safe_load((args.reference_run / 'train_config.yml').read_text())
    config = argparse.Namespace(**saved['routing'])
    config.encoder_method = args.method
    config.encoder_ckpt = args.checkpoint
    config.encoder_size = 'official' if external else 'S'
    config.encoder_pretrained_mode = 'tactile' if external else 'univtac'
    config.encoder_pretrain_init = 'adapted' if external else 'scratch'
    config.tactile_type = 'pool3x3'
    config.output_dir = args.output
    # One GPU with the same 16 samples/GPU as the four-GPU, global-64 DLC job.
    config.batch_size = 16
    config.steps = 2
    config.workers = 2
    config.log_freq = 1
    config.save_freq = 2
    config.save_checkpoint = False
    config.max_val_batches = 1
    config.device = 'cuda'
    config.profile_stages = False
    config.wan_training_optimization = 'compiled'
    report = dict(method=args.method, global_batch=16, world_size=1,
                  dlc_global_batch=64, dlc_world_size=4, samples_per_gpu=16,
                  encoder_gradient_finite=True, encoder_gradient_nonzero=False)
    if not external:
        report.update(conv3d_gradient_finite=True, conv3d_gradient_nonzero=False)
    original_init = TactileEncoder.__init__

    def checked_init(self, *positional, **keywords):
        original_init(self, *positional, **keywords)
        assert self.num_frames == 4
        assert self.mode == 'pool3x3'
        if external:
            convolution = next(p for name, p in self.backbone.named_parameters()
                               if p.ndim in (4, 5) and ('patch_embed' in name or 'patch_embedding' in name))
        else:
            assert self.backbone.output_frames == 2 and self.backbone.temporal_compression == '4in2'
            convolution = next(p for p in self.backbone.parameters() if p.ndim == 5)
        assert convolution.requires_grad
        report['projection_in_features'] = self.proj.in_features
        if args.method == 'SD-VAE4Frame':
            assert self.backbone.sd_patch_size == 2 and self.proj.in_features == 16

        def gradient_hook(gradient):
            finite = bool(torch.isfinite(gradient).all())
            nonzero = bool(gradient.abs().sum() > 0)
            report['encoder_gradient_finite'] &= finite
            report['encoder_gradient_nonzero'] |= nonzero
            if not external:
                report['conv3d_gradient_finite'] &= finite
                report['conv3d_gradient_nonzero'] |= nonzero

        convolution.register_hook(gradient_hook)

    TactileEncoder.__init__ = checked_init
    original_forward = TactileEncoder.forward_flat

    def checked_forward(self, batch):
        output = original_forward(self, batch)
        assert output.shape[1] == tokens_per_sensor * len(self.keys), output.shape
        report['tactile_tokens'] = output.shape[1]
        report['tokens_per_sensor'] = tokens_per_sensor
        return output

    TactileEncoder.forward_flat = checked_forward
    episodes = json.loads((args.reference_run / 'episodes.json').read_text())['train']
    stats = json.loads((args.reference_run / 'dataset_stats.json').read_text())

    def verified_stats(files, action_mode, chunk_size):
        assert {task: [p.name for p in paths] for task, paths in files.items()} == episodes
        assert action_mode == saved['action_mode'] and chunk_size == saved['chunk_size']
        return stats

    trainer.compute_stats = verified_stats
    torch.set_num_threads(4)
    try:
        trainer.setup_distributed(config)
        with trainer.training_log(config):
            trainer.run(config)
    finally:
        if trainer.dist.is_initialized():
            trainer.dist.destroy_process_group()
    text = (args.output / 'train.log').read_text().splitlines()
    training = [line for line in text if line.startswith('step=2 loss=')]
    validation = [line for line in text if line.startswith('step=2 val_loss=')]
    assert training and validation, 'Training or validation did not complete'
    for line in training + validation:
        value = next(word.split('=', 1)[1] for word in line.split()
                     if word.startswith(('loss=', 'val_loss=')))
        assert math.isfinite(float(value)), line
    assert report['encoder_gradient_finite'] and report['encoder_gradient_nonzero'], report
    report.update(training=training[-1], validation=validation[-1], passed=True)
    temporary = args.output / 'validation_passed.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output / 'validation_passed.json')
    print('TEMPORAL_STARVLA_VALIDATED ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
