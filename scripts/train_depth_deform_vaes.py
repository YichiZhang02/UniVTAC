"""Train selected supported depth/deform RVAE-S modes on available GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = tuple(f'depth_deform_{n}' for n in (20, 80, 160, 240)) + (
    'depth_deform_nonlinear', 'depth_deform_nonlinear_radial')


def train(mode, gpu, run_id, epochs, patience):
    output = ROOT / 'encoder_results' / mode / 'univtac' / 'RVAE' / 'S' / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHON_BIN=sys.executable)
    command = ['bash', str(ROOT / 'train_encoder.sh'), '--method', 'RVAE',
               '--size', 'S', '--input', mode, '--', '--epochs', str(epochs),
               '--early-stop-patience', str(patience), '--checkpoint-metric', 'val_loss',
               '--device', 'cuda', '--output-dir', str(output)]
    status = dict(method='RVAE', size='S', input_mode=mode, gpu=gpu,
                  state='running', output=str(output), command=command,
                  epochs=epochs, early_stop_patience=patience)
    started = time.time()
    with (output / 'driver.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        status['pid'] = process.pid
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
        code = process.wait()
    status.update(state='completed' if code == 0 else 'failed', exit_code=code,
                  seconds=round(time.time() - started, 1))
    metrics = output / 'metrics.jsonl'
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        validation = [row for row in rows if 'val_loss' in row]
        if validation:
            best = min(validation, key=lambda row: row['val_loss'])
            status.update(epochs_completed=validation[-1]['epoch'],
                          best_epoch=best['epoch'], best_val_loss=best['val_loss'])
    (output / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--early-stop-patience', type=int, default=3)
    parser.add_argument('--gpus', default='1,0', help='Round-robin physical GPU IDs')
    parser.add_argument('--inputs', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--run-id', default=datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    args = parser.parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    if args.epochs < 1 or args.early_stop_patience < 0 or not gpus:
        parser.error('Epochs/GPU list must be positive/nonempty; early-stop patience must be >=0')
    if len(set(args.inputs)) != len(args.inputs):
        parser.error('Input modes must be distinct')
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(train, mode, gpus[index % len(gpus)], args.run_id,
                            args.epochs, args.early_stop_patience)
                for index, mode in enumerate(args.inputs)]
        failed = any([job.result() != 0 for job in jobs])
    raise SystemExit(int(failed))
