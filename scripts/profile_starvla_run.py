"""Profile an existing StarVLA run with the same data and encoder contract.

Example: torchrun --standalone --nproc-per-node=2 scripts/profile_starvla_run.py
  --run-config policy_results/<run>/train_config.yml --output /tmp/profile_mae

Batch size is PER GPU. Saved dataset statistics are reused only after verifying
the exact training episode split. Model weights start from the original model
and encoder initializers; this measures performance, not checkpoint convergence.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
import policy.train_unified as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-per-gpu', type=int, default=16)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--preprocessing', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--wan-training-optimization', choices=['reference', 'compiled'], default='compiled')
    cli = parser.parse_args()
    if not 0 <= cli.warmup < cli.steps:
        parser.error('warmup must be nonnegative and smaller than steps')
    saved = yaml.safe_load(cli.run_config.read_text())
    args = argparse.Namespace(**saved['routing'])
    args.output_dir = cli.output
    args.encoder_ckpt = Path(args.encoder_ckpt)
    args.encoder_pretrain_init = getattr(args, 'encoder_pretrain_init', None)
    args.steps, args.workers = cli.steps, cli.workers
    args.device = 'cuda'
    args.batch_size = cli.batch_per_gpu * int(trainer.os.environ.get('WORLD_SIZE', '1'))
    args.log_freq, args.save_freq = 5, cli.steps
    args.save_checkpoint, args.max_val_batches = False, 0
    args.tactile_preprocessing = cli.preprocessing
    args.wan_training_optimization = cli.wan_training_optimization
    args.profile_stages, args.profile_warmup = True, cli.warmup
    episodes = json.loads(cli.run_config.with_name('episodes.json').read_text())['train']
    stats = json.loads(cli.run_config.with_name('dataset_stats.json').read_text())

    def saved_stats(files, action_mode, chunk_size):
        actual = {task: [path.name for path in paths] for task, paths in files.items()}
        if actual != episodes or action_mode != saved['action_mode'] or chunk_size != saved['chunk_size']:
            raise ValueError('Saved dataset statistics do not match requested training split')
        return stats

    trainer.compute_stats = saved_stats
    torch.set_num_threads(2)
    start = time.perf_counter()
    try:
        trainer.setup_distributed(args)
        with trainer.training_log(args):
            trainer.run(args)
        if args.rank == 0:
            metadata = dict(source_config=str(cli.run_config), batch_per_gpu=cli.batch_per_gpu,
                            world_size=args.world_size, global_batch=args.batch_size,
                            steps=cli.steps, warmup=cli.warmup, workers_per_rank=cli.workers,
                            preprocessing=cli.preprocessing, dataset_statistics='verified saved split',
                            save_checkpoint=False, validation_batches=0,
                            total_run_seconds=time.perf_counter()-start)
            (cli.output / 'profile_metadata.json').write_text(json.dumps(metadata, indent=2))
    finally:
        if trainer.dist.is_initialized():
            trainer.dist.destroy_process_group()


if __name__ == '__main__':
    main()
