"""Hourly adapted DLC encoder completion -> local policy test -> StarVLA DLC."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_encoder_dlc import PYTHON, remote
from scripts.watch_encoder_starvla import atomic_json, utc
from scripts.watch_temporal_starvla import TemporalPolicyMonitor, process_key, run_local_validation

METHODS = ('Sparsh_mae', 'Sparsh_DINOv2', 'Sparsh_IJEPA', 'Sparsh_VJEPA', 'Anytouch2')


def parse_blocks(path):
    blocks = {}
    for paragraph in path.read_text().split('\n\n'):
        if not re.search(r'^ENCODER_METHOD=', paragraph, re.M):
            continue
        lines = [line.rstrip(' \\') for line in paragraph.splitlines()
                 if line and not line.lstrip().startswith('#')]
        tokens = shlex.split(' '.join(lines))
        split = tokens.index('bash')
        environment = dict(token.split('=', 1) for token in tokens[:split])
        method = environment['ENCODER_METHOD']
        if method not in METHODS or method in blocks:
            raise ValueError('Duplicate or unsupported adapted encoder')
        expected = dict(CUDA_VISIBLE_DEVICES='0,1,2,3', NUM_PROCESSES='4',
                        TACTILE_MODE='encode', TACTILE_TYPE='pool3x3', TACTILE_INSERT_LOCATION='encoder',
                        TACTILE_INPUT_MODE='depth_deform_nonlinear_radial', ENCODER_PRETRAINED_MODE='tactile',
                        ENCODER_PRETRAIN_INIT='adapted', ENCODER_INIT='latest', ENCODER_SIZE='official')
        if any(environment.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{method}: unexpected policy settings')
        if tokens[split:] != ['bash', str(ROOT / 'training/train_starvla_groot.sh')]:
            raise ValueError('Expected the StarVLA-GR00T launcher')
        blocks[method] = environment
    if set(blocks) != set(METHODS):
        raise ValueError('Expected all five adapted encoders in 3.sh')
    return blocks


def encoder_status(record, remote_call=remote):
    rows = json.loads(Path(record['encoder_ledger']).read_text())
    row = next(row for row in rows if row['method'] == record['method'])
    if (Path(row['output_dir']) / 'encoder.pth').resolve() != Path(record['source_checkpoint']).resolve():
        raise ValueError('Adaptation run no longer matches the pinned checkpoint')
    if row.get('initialization') != 'adapted':
        raise ValueError('Encoder job must use adapted initialization')
    if not row.get('job_id'):
        return 'Running', {'submission_status': row['status']}
    record['encoder_job_id'] = row['job_id']
    code, output = remote_call(['get', 'job', row['job_id'], '--show_detail'])
    if code:
        raise RuntimeError(output[-1000:])
    detail = json.loads(output)
    if detail.get('UserCommand') is not None and detail['UserCommand'] != row['command']:
        raise ValueError('DLC encoder command differs from the recorded adaptation command')
    status = detail['Status']
    return ('Failed' if status in ('Failed', 'Stopped', 'Terminated') else status), detail


def validate_checkpoint(path, record):
    code = '''import json,sys,torch
s=torch.load(sys.argv[1],map_location='cpu',weights_only=True,mmap=True)
assert s.get('encoder'), 'Missing encoder weights'
keys=('method','size','input_mode','initialization','pretrained_mode','color_order',
      'num_frames','frame_stride','image_size','epoch','step','val_loss','source_revision',
      'training_objective','mae_mask_ratio','mae_decoder_init')
print(json.dumps({k:s.get(k) for k in keys}))
'''
    result = subprocess.run([PYTHON, '-c', code, str(path)], text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    metadata = json.loads(result.stdout)
    expected = dict(method=record['method'], size='official', input_mode='depth_deform_nonlinear_radial',
                    initialization='adapted', pretrained_mode='tactile', color_order='RGB',
                    num_frames=4, frame_stride=2, image_size=[224, 224])
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'{key}={metadata.get(key)!r}, expected {value!r}')
    if record['method'] == 'Anytouch2':
        if (metadata.get('training_objective') != 'video_mae_masked_pixels_v1'
                or metadata.get('mae_mask_ratio') != .5
                or metadata.get('mae_decoder_init') != 'random'):
            raise ValueError('AnyTouch2 requires the corrected video-MAE checkpoint')
    if not (metadata['source_revision'] and 1 <= metadata['epoch'] <= 20 and
            metadata['step'] > 0 and math.isfinite(metadata['val_loss'])):
        raise ValueError('Invalid adaptation checkpoint provenance/progress')
    return metadata


class AdaptedPolicyMonitor(TemporalPolicyMonitor):
    def __init__(self, folder, remote_call=remote, validator=validate_checkpoint,
                 local_validator=run_local_validation):
        super().__init__(folder, remote_call=remote_call, validator=validator,
                         readiness=lambda record: encoder_status(record, remote_call),
                         local_validator=local_validator)


def initialize(folder, blocks_file, ledger, scheduler_status, interval):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'state.json').exists():
        return json.loads((folder / 'state.json').read_text())
    blocks = parse_blocks(blocks_file)
    rows = {row['method']: row for row in json.loads(ledger.read_text())}
    for method, environment in blocks.items():
        if Path(environment['ENCODER_CKPT']).resolve() != (Path(rows[method]['output_dir']) / 'encoder.pth').resolve():
            raise ValueError(f'{method}: 3.sh does not match the adaptation ledger')
    source = folder / 'source'
    ignore = shutil.ignore_patterns('__pycache__', '*.ipynb', '.git')
    shutil.copytree(ledger.parent / 'source/encoder', source / 'encoder', ignore=ignore)
    (source / 'policy').mkdir()
    for item in (ROOT / 'policy').iterdir():
        if item.is_file():
            shutil.copy2(item, source / 'policy' / item.name)
    for name in ('vla_common', 'starvla_groot', 'starvla_pi', 'pi05', 'n0vtla'):
        shutil.copytree(ROOT / 'policy' / name, source / 'policy' / name, ignore=ignore)
    shutil.copytree(ROOT / 'training', source / 'training', ignore=ignore)
    shutil.copy2(ROOT / 'train_policy.sh', source / 'train_policy.sh')
    (source / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/local_temporal_starvla_smoke.py', source / 'scripts/local_temporal_starvla_smoke.py')
    for name in ('resources', 'encoder_results', 'policy_results'):
        (source / name).symlink_to(ROOT / name, target_is_directory=True)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for top in ('encoder', 'policy', 'training', 'scripts') for p in (source / top).rglob('*')
              if p.is_file()}
    hashes['train_policy.sh'] = hashlib.sha256((source / 'train_policy.sh').read_bytes()).hexdigest()
    atomic_json(folder / 'source_sha256.json', hashes)
    shutil.copy2(blocks_file, folder / '3.sh')
    reference = folder / 'reference'
    reference.mkdir()
    prior = json.loads((ROOT / '.cache/temporal_starvla_monitor_current.json').read_text())
    for name in ('train_config.yml', 'episodes.json', 'dataset_stats.json'):
        shutil.copy2(Path(prior['state_dir']) / 'reference' / name, reference / name)
    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    records = []
    for method, environment in blocks.items():
        environment.update(WORKERS='2', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                           NCCL_DEBUG='WARN', NCCL_SOCKET_IFNAME='eth0')
        records.append(dict(method=method, environment=environment, phase='WaitingEncoder',
            name=f'yichi_adapted_{stamp}_{method.lower().replace("-", "")}_pool3x3',
            encoder_job_id=rows[method].get('job_id', 'pending:'+method), encoder_ledger=str(ledger),
            gpu_scheduler_status_file=str(scheduler_status),
            source_checkpoint=str(Path(environment['ENCODER_CKPT']).resolve()),
            frozen_checkpoint=str(folder / 'checkpoints' / method / 'encoder.pth'),
            output_dir=str(folder / 'runs' / method)))
    state = dict(status='Active', created_at=utc(), interval_seconds=interval, records=records,
                 source_root=str(source), encoder_ledger=str(ledger), history=[], reference_run=str(reference),
                 source_blocks=str(blocks_file), source_blocks_sha256=hashlib.sha256(blocks_file.read_bytes()).hexdigest())
    atomic_json(folder / 'state.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('start', 'run', 'once', 'status', 'stop'):
        action.add_argument('--'+name, action='store_true')
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--encoder-ledger', type=Path)
    parser.add_argument('--gpu-scheduler-status', type=Path)
    parser.add_argument('--blocks', type=Path, default=ROOT / '3.sh')
    parser.add_argument('--interval', type=int, default=3600)
    args = parser.parse_args()
    folder = args.state_dir.resolve()
    if args.status:
        print((folder / 'state.json').read_text())
        return
    if args.stop:
        pid = int((folder / 'monitor.pid').read_text())
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        if 'watch_adapted_starvla.py' in command and str(folder) in command:
            os.kill(pid, signal.SIGTERM)
        return
    if not args.encoder_ledger or not args.gpu_scheduler_status or args.interval < 1:
        parser.error('Encoder ledger, GPU scheduler status and a positive interval are required')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        initialize(folder, args.blocks.resolve(), args.encoder_ledger.resolve(),
                   args.gpu_scheduler_status.resolve(), args.interval)
    if args.start:
        pid_file = folder / 'monitor.pid'
        if pid_file.exists() and process_key(int(pid_file.read_text())):
            raise RuntimeError('A monitor already exists for this state directory')
        with (folder / 'monitor.log').open('a') as log:
            child = subprocess.Popen([PYTHON, '-u', str(Path(__file__).resolve()), '--run',
                '--state-dir', str(folder), '--encoder-ledger', str(args.encoder_ledger.resolve()),
                '--gpu-scheduler-status', str(args.gpu_scheduler_status.resolve()),
                '--blocks', str(args.blocks.resolve()), '--interval', str(args.interval)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Monitor PID {child.pid}; state {folder / "state.json"}', flush=True)
        return
    with (folder / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (folder / 'monitor.pid').write_text(str(os.getpid()))
        monitor = AdaptedPolicyMonitor(folder)
        while True:
            state = monitor.tick()
            monitor.log(str({r['method']: r['phase'] for r in state['records']}))
            if args.once or state['status'] == 'Complete':
                return
            deadline = time.monotonic()+state['interval_seconds']
            while time.monotonic() < deadline:
                time.sleep(min(60, deadline-time.monotonic()))


if __name__ == '__main__':
    main()
