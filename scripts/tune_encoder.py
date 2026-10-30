"""Compare learning rates at fixed batch size 64 on a fixed file-level split."""
import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from encoder.catalog import LOCAL_METHODS
METHODS = LOCAL_METHODS
INPUTS = ('marker_only', 'rgb_only', 'marker_rgb', 'depth_deform_20', 'depth_deform_80', 'depth_deform_160', 'depth_deform_240', 'depth_deform_nonlinear', 'depth_deform_nonlinear_radial')


def run_trial(case, args, gpu):
    method, size, input_mode, batch, lr = case
    run_id = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT / 'encoder_results' / input_mode / 'univtac' / method / size / run_id
    command = [str(ROOT / 'train_encoder.sh'), '--method', method, '--size', size,
               '--input', input_mode, '--data-root', str(args.data_root), '--',
               '--samples', str(args.samples), '--epochs', str(args.epochs),
               '--batch-size', str(batch), '--lr', str(lr), '--workers', str(args.workers),
               '--output-dir', str(output), '--no-save']
    if method in ('DINOv2', 'I-JEPA', 'V-JEPA'):
        command += ['--probe-per-class-train', '32', '--probe-per-class-val', '16']
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'driver.log').open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    metrics_file = output / 'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_file.read_text().splitlines()] if metrics_file.exists() else []
    validation = [record for record in metrics if 'val_loss' in record]
    training = [record for record in metrics if 'loss' in record]
    score = 'knn_accuracy' if method in ('DINOv2', 'I-JEPA', 'V-JEPA') else 'val_loss'
    best = (max(validation, key=lambda record: record.get(score, float('-inf')))
            if score == 'knn_accuracy' else min(validation, key=lambda record: record['val_loss'])) if validation else {}
    row = {'method': method, 'size': size, 'input_mode': input_mode,
           'batch': batch, 'lr': lr, 'gpu': gpu,
           'exit_code': result.returncode, 'seconds': round(time.time() - started, 1),
           'best_epoch': best.get('epoch'), 'best_val_loss': best.get('val_loss'),
           'best_knn_accuracy': best.get('knn_accuracy'),
           'final_val_loss': validation[-1]['val_loss'] if validation else None,
           'cls_target_entropy': training[-1].get('cls_target_entropy') if training else None,
           'target_feature_std': training[-1].get('target_feature_std') if training else None,
           'val_cls_target_entropy': validation[-1].get('val_cls_target_entropy') if validation else None,
           'val_target_feature_std': validation[-1].get('val_target_feature_std') if validation else None,
           'output': str(output)}
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    parser.add_argument('--batch-results', type=Path, nargs='+', help='Legacy argument; batch is always 64')
    parser.add_argument('--batch-source', choices=('compare', 'config'), default='config', help='Legacy argument; batch is always 64')
    parser.add_argument('--inputs', default='marker_rgb')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--scales', default='0.3,1,2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for method in args.methods.split(','):
        settings = yaml.safe_load((ROOT / 'encoder' / method / 'config.yml').read_text())
        for size in ('S', 'B'):
            batches = [64]
            for input_mode in args.inputs.split(','):
                if input_mode not in INPUTS:
                    parser.error(f'Unknown input mode: {input_mode}')
                for batch in batches:
                    for scale in (float(value) for value in args.scales.split(',')):
                        cases.append((method, size, input_mode, batch, settings[size]['lr'] * scale))
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    buckets = [cases[i::len(gpus)] for i in range(len(gpus))]
    def worker(gpu, bucket):
        return [run_trial(case, args, gpu) for case in bucket]
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, bucket)
                   for gpu, bucket in zip(gpus, buckets)]
        rows = [row for future in as_completed(futures) for row in future.result()]
    with (args.output / 'trials.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row['method'], row['size'], row['input_mode'], row['batch'], row['lr'])))
    if any(row['exit_code'] for row in rows):
        raise SystemExit('Some tuning trials failed; check the corresponding train.log')


if __name__ == '__main__':
    main()
