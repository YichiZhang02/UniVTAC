"""Run the complete encoder architecture/input smoke grid on available GPUs."""
import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ResNet', 'VAE', 'MAE', 'DINOv2', 'I-JEPA', 'V-JEPA')
INPUTS = ('marker_only', 'rgb_only', 'marker_rgb', 'depth_deform')


def run_case(case, args, gpu):
    method, size, input_mode = case
    name = f'{method}_{size}_{input_mode}'
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / 'train_encoder.sh'), '--method', method, '--size', size,
               '--input', input_mode, '--data-root', str(args.data_root), '--',
               '--samples', str(args.samples), '--epochs', str(args.epochs),
               '--batch-size', str(args.batch_size), '--workers', str(args.workers),
               '--max-steps', str(args.max_steps), '--max-val-steps', str(args.max_val_steps),
               '--output-dir', str(output), '--no-save']
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    start = time.time()
    with (output / 'driver.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    metrics_path = output / 'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    train = [record for record in metrics if 'loss' in record]
    val = [record for record in metrics if 'val_loss' in record]
    row = dict(method=method, size=size, input_mode=input_mode, gpu=gpu,
               exit_code=result.returncode, seconds=round(time.time() - start, 1),
               train_loss=train[-1]['loss'] if train else '',
               val_loss=val[-1]['val_loss'] if val else '',
               step_seconds=train[-1]['step_seconds'] if train else '',
               batch_seconds=train[-1].get('batch_seconds', '') if train else '',
               actual_batch_size=train[-1].get('batch_size', '') if train else '',
               samples_per_second=train[-1].get('samples_per_second', '') if train else '',
               gpu_peak_gb=train[-1].get('gpu_peak_gb', '') if train else '')
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--sizes', default='S,B')
    parser.add_argument('--inputs', default=','.join(INPUTS))
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=2)
    parser.add_argument('--max-val-steps', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    cases = [(method, size, input_mode) for method in args.methods.split(',')
             for size in args.sizes.split(',') for input_mode in args.inputs.split(',')]
    buckets = [cases[i::len(gpus)] for i in range(len(gpus))]

    def worker(gpu, bucket):
        return [run_case(case, args, gpu) for case in bucket]

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, bucket)
                   for gpu, bucket in zip(gpus, buckets)]
        rows = [row for future in as_completed(futures) for row in future.result()]
    with (args.output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row['method'], row['size'], row['input_mode'])))
    if any(row['exit_code'] for row in rows):
        raise SystemExit('One or more encoder settings failed; see driver.log in each run directory')


if __name__ == '__main__':
    main()
