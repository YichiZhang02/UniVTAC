"""Train every encoder method, size, and tactile input on two single-GPU workers."""
import argparse
import csv
import json
import os
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ResNet', 'VAE', 'MAE', 'DINOv2', 'I-JEPA', 'V-JEPA')
INPUTS = ('marker_rgb', 'marker_only', 'rgb_only', 'depth_deform')


def train(case, gpu, data_root):
    method, size, input_mode = case
    run_id = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT / 'encoder_results' / input_mode / method / size / run_id
    command = [str(ROOT / 'train_encoder.sh'), '--method', method, '--size', size,
               '--input', input_mode, '--data-root', str(data_root), '--',
               '--output-dir', str(output)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'driver.log').open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    metrics_file = output / 'metrics.jsonl'
    records = [json.loads(line) for line in metrics_file.read_text().splitlines()] if metrics_file.exists() else []
    validation = [record for record in records if 'val_loss' in record]
    metric = 'knn_accuracy' if method in ('DINOv2', 'I-JEPA', 'V-JEPA') else 'val_loss'
    best = (max(validation, key=lambda record: record.get(metric, float('-inf')))
            if metric == 'knn_accuracy' else min(validation, key=lambda record: record['val_loss'])) if validation else {}
    row = {'method': method, 'size': size, 'input_mode': input_mode, 'gpu': gpu,
           'exit_code': result.returncode, 'seconds': round(time.time() - started, 1),
           'epochs_completed': validation[-1]['epoch'] if validation else 0,
           'best_epoch': best.get('epoch'), 'best_val_loss': best.get('val_loss'),
           'best_knn_accuracy': best.get('knn_accuracy'),
           'output': str(output)}
    (output / 'status.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    parser.add_argument('--output', type=Path, required=True,
                        help='Directory for the cross-run summary; checkpoints use encoder_results')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--inputs', default=','.join(INPUTS))
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--sizes', default='S,B')
    parser.add_argument('--skip-completed', action='store_true',
                        help='Reuse successful status.json files from earlier grid runs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [(method, size, input_mode) for input_mode in args.inputs.split(',')
             for method in args.methods.split(',') for size in args.sizes.split(',')]
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    pending = queue.Queue()
    for case in cases:
        pending.put(case)

    def worker(gpu):
        rows = []
        while True:
            try:
                case = pending.get_nowait()
            except queue.Empty:
                return rows
            if args.skip_completed:
                method, size, input_mode = case
                base = ROOT / 'encoder_results' / input_mode / method / size
                completed = []
                for status_path in base.glob('*/status.json'):
                    status = json.loads(status_path.read_text())
                    if status.get('exit_code') == 0:
                        completed.append((status_path.stat().st_mtime, status))
                if completed:
                    row = max(completed, key=lambda item: item[0])[1]
                    print(json.dumps({'skipped_completed': True, **row}), flush=True)
                    rows.append(row)
                    continue
            rows.append(train(case, gpu, args.data_root))

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        rows = [row for future in as_completed(futures) for row in future.result()]
    with (args.output / 'summary.csv').open('w', newline='') as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row['input_mode'], row['method'], row['size'])))
    if any(row['exit_code'] for row in rows):
        raise SystemExit('Some formal runs failed; see their train.log and status.json')


if __name__ == '__main__':
    main()
