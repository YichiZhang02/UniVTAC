"""Run all eight scratch encoders with 4in2 by default, one process per GPU.

The status file is updated after every launch/completion. Smoke mode performs a
real global-batch optimizer step, validation, and checkpoint save for each model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'encoder')]
from encoder.catalog import MULTIFRAME_METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--input-mode', default='depth_deform_nonlinear_radial')
    parser.add_argument('--size', choices=['S', 'B'], default='S')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early-stop-patience', type=int, default=3)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--micro-batch-size', type=int, default=8)
    parser.add_argument('--methods', nargs='+', choices=MULTIFRAME_METHODS, default=list(MULTIFRAME_METHODS))
    parser.add_argument('--compressions', nargs='+', choices=['4in1', '4in2'], default=['4in2'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight-status', type=Path, help='Require matching completed smoke jobs before training')
    parser.add_argument('--defer-gpus', default='', help='GPUs reserved for preflight until all smoke jobs finish')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Frozen source tree used for encoder training subprocesses')
    args = parser.parse_args()
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output = args.output_dir or ROOT / ('.cache/temporal_validation/smoke' if args.smoke else 'encoder_results/multiframe_training') / stamp
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gpu_ids = args.gpus.split(',')
    jobs = []
    for mode in args.compressions:
        for name in args.methods:
            run_dir = (output / 'runs' / name / mode if args.smoke else
                       ROOT / 'encoder_results' / args.input_mode / 'univtac' / name / args.size / 'scratch' / mode / stamp)
            jobs.append(dict(method=name, compression=mode, output_dir=str(run_dir), state='queued'))
    lock = threading.Lock()
    status = dict(pid=os.getpid(), started=stamp, smoke=args.smoke, input_mode=args.input_mode,
                  temporal_backend='conv3d',
                  size=args.size, num_frames=4, frame_stride=1, simulation_step_stride=2, global_batch_size=64,
                  micro_batch_size=args.micro_batch_size, samples=256 if args.smoke else args.samples,
                  epochs=1 if args.smoke else args.epochs,
                  early_stop_patience=0 if args.smoke else args.early_stop_patience, jobs=jobs)

    def save():
        # Writes from worker threads are serialized; replace is atomic for readers.
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(output / 'status.json')

    save()

    def worker(gpu):
        while True:
            ready = None
            if args.preflight_status:
                try:
                    preflight = json.loads(args.preflight_status.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(5)
                    continue
                for key, expected in [('input_mode', args.input_mode), ('size', args.size),
                                      ('temporal_backend', 'conv3d'),
                                      ('num_frames', 4), ('frame_stride', 1), ('simulation_step_stride', 2),
                                      ('micro_batch_size', args.micro_batch_size)]:
                    if preflight.get(key) != expected:
                        raise ValueError(f'Preflight {key}={preflight.get(key)!r}, expected {expected!r}')
                if gpu in args.defer_gpus.split(',') and not preflight.get('finished'):
                    time.sleep(5)
                    continue
                if any(job['state'] == 'failed' for job in preflight['jobs']):
                    raise RuntimeError('Smoke preflight failed; inspect preflight status before training')
                ready = {(job['method'], job['compression']) for job in preflight['jobs'] if job['state'] == 'complete'}
            with lock:
                job = next((job for job in jobs if job['state'] == 'queued' and
                            (ready is None or (job['method'], job['compression']) in ready)), None)
                if job is None and not any(job['state'] == 'queued' for job in jobs):
                    return
                wait_for_preflight = job is None
                if job is not None:
                    job.update(state='starting', gpu=gpu)
                    save()
            if wait_for_preflight:
                time.sleep(5)
                continue
            run_dir = Path(job['output_dir'])
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(args.source_root.resolve() / 'encoder' / job['method'] / 'train.py'),
                       '--size', args.size, '--input-mode', args.input_mode,
                       '--temporal-compression', job['compression'], '--workers', str(args.workers),
                       '--micro-batch-size', str(args.micro_batch_size),
                       '--epochs', str(1 if args.smoke else args.epochs),
                       '--samples', str(256 if args.smoke else args.samples),
                       '--early-stop-patience', str(0 if args.smoke else args.early_stop_patience),
                       '--checkpoint-metric', 'val_loss',
                       '--output-dir', str(run_dir)]
            if args.smoke:
                command.extend(['--max-steps', '1', '--max-val-steps', '1'])
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            with (run_dir / 'train.log').open('a') as log:
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    with lock:
                        job.update(state='running', pid=process.pid, command=command)
                        save()
                    print(f"START gpu={gpu} {job['method']} {job['compression']} pid={process.pid}", flush=True)
                    code = process.wait()
                    with lock:
                        job.update(state='complete' if code == 0 else 'failed', returncode=code)
                        save()
                    print(f"END {job['method']} {job['compression']} code={code}", flush=True)
                except Exception as error:
                    with lock:
                        job.update(state='failed', error=repr(error))
                        save()
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        list(pool.map(worker, gpu_ids))
    status['finished'] = datetime.now().isoformat()
    save()
    return int(any(job['state'] != 'complete' for job in jobs))


if __name__ == '__main__':
    raise SystemExit(main())
