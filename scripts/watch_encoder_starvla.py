"""Hourly encoder completion checks followed by the matching 2.sh DLC submission.

Start with --start, inspect with --status, and stop with --stop. The monitor
checks immediately, then hourly; it exits once all five policies are submitted.
Only successful encoder jobs are eligible, including successful early stopping.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

if __package__:
    from .submit_encoder_dlc import COMMON, PYTHON, ROOT, remote
else:
    from submit_encoder_dlc import COMMON, PYTHON, ROOT, remote

sys.path.insert(0, str(ROOT))
from encoder.catalog import canonical_name, resolve_identity

DEFAULT_LEDGER = ROOT / 'encoder_results/dlc_encoder_gpu_tps_5epochs_20261026/submissions.json'
DEFAULT_STATE = ROOT / 'policy_results/encoder_to_starvla_20261026'
RESOURCES = ['--worker_gpu=4', '--worker_cpu=16', '--worker_memory=128Gi', '--worker_shared_memory=32Gi']


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_blocks(path):
    """Read the five independently copyable env/bash blocks without eval."""
    blocks = {}
    for paragraph in path.read_text().split('\n\n'):
        lines = [line.rstrip(' \\') for line in paragraph.splitlines()
                 if line and not line.lstrip().startswith('#')]
        if not any(re.search(r'(^|\s)ENCODER_METHOD=', line) for line in lines):
            continue
        tokens = shlex.split(' '.join(lines))
        split = tokens.index('bash')
        environment = dict(token.split('=', 1) for token in tokens[:split])
        if tokens[split:] != ['bash', str(ROOT / 'training/train_starvla_groot.sh')]:
            raise ValueError('Expected the StarVLA-GR00T launcher from 2.sh')
        if (environment.get('ENCODER_PRETRAIN_INIT') != 'scratch'
                or environment.get('ENCODER_INIT') != 'latest'
                or environment.get('NUM_PROCESSES') != '4'
                or environment.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3'):
            raise ValueError('2.sh must explicitly select scratch checkpoints and four-GPU training')
        method = environment['ENCODER_METHOD']
        if method in blocks:
            raise ValueError(f'Duplicate encoder block: {method}')
        blocks[method] = environment
    if {canonical_name(env['ENCODER_PRETRAINED_MODE'], method) for method, env in blocks.items()} != {'Sparsh_mae', 'Sparsh_DINOv2', 'Sparsh_IJEPA', 'Sparsh_VJEPA', 'Wan22_VAE'}:
        raise ValueError('Expected the four Sparsh models and Wan2.2-VAE in 2.sh')
    return blocks


def validate_checkpoint(path, record):
    # Read only archive metadata/tensor descriptors, without loading every
    # tensor into RAM. A truncated archive fails torch.load's integrity checks.
    code = '''import json,sys,torch
s=torch.load(sys.argv[1],map_location="cpu",weights_only=True,mmap=True)
assert isinstance(s.get("encoder"),dict) and s["encoder"], "Missing encoder tensors"
keys=("initialization","pretrained_mode","method","size","input_mode","color_order","epoch","step","val_loss","num_frames")
print(json.dumps({k:s.get(k) for k in keys}))
'''
    result = subprocess.run([PYTHON, '-c', code, str(path)], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'Invalid encoder checkpoint {path}: {result.stderr[-1200:]}')
    metadata = json.loads(result.stdout)
    expected = {'initialization': 'scratch', 'pretrained_mode': record['environment']['ENCODER_PRETRAINED_MODE'],
                'method': record['method'], 'size': record['environment']['ENCODER_SIZE'],
                'input_mode': record['environment']['TACTILE_INPUT_MODE'], 'color_order': 'RGB'}
    mode = expected['pretrained_mode']
    if resolve_identity(mode, metadata.get('method'), metadata.get('size')) != resolve_identity(mode, expected['method'], expected['size']):
        raise ValueError(f'{path}: encoder identity does not match requested model')
    for key, value in expected.items():
        if key in ('method', 'size'):
            continue
        if metadata.get(key) != value:
            raise ValueError(f'{path}: {key}={metadata.get(key)!r}, expected {value!r}')
    return metadata


def freeze_checkpoint(source, target):
    """Encoder files are published by atomic rename; a hardlink pins the inode."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        temporary = target.with_suffix('.copying')
        shutil.copy2(source, temporary)
        temporary.replace(target)


class Monitor:
    def __init__(self, folder, remote_call=remote, validator=validate_checkpoint):
        self.folder = Path(folder)
        self.path = self.folder / 'state.json'
        self.remote = remote_call
        self.validate = validator

    def save(self, state):
        state['updated_at'] = utc()
        atomic_json(self.path, state)

    def log(self, message):
        print(f'{utc()} {message}', flush=True)

    def initialize(self, blocks_file, encoder_ledger, interval):
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state['interval_seconds'] != interval:
                raise ValueError('Existing monitor has a different interval')
            return state
        blocks = parse_blocks(blocks_file)
        encoders = {row['method']: row for row in json.loads(encoder_ledger.read_text())}
        records = []
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        for method, environment in blocks.items():
            name = canonical_name(environment['ENCODER_PRETRAINED_MODE'], method)
            encoder = encoders.get(method) or encoders[name]
            source = Path(environment['ENCODER_CKPT']).resolve()
            if source != (Path(encoder['output_dir']) / 'encoder.pth').resolve():
                raise ValueError(f'{method}: 2.sh checkpoint does not match the monitored encoder job')
            key = method.lower().replace('.', '').replace('-', '')
            records.append(dict(method=method, encoder_job_id=encoder['job_id'],
                source_checkpoint=str(source), environment=environment, phase='WaitingEncoder',
                name=f'yichi_train_{stamp}_{key}_scratch_ft',
                output_dir=str(self.folder / 'runs' / key),
                frozen_checkpoint=str(self.folder / 'checkpoints' / key / 'encoder.pth')))
        shutil.copy2(blocks_file, self.folder / '2.sh')
        state = dict(version=1, monitor_id=uuid.uuid4().hex, created_at=utc(),
                     source_blocks=str(blocks_file), source_blocks_sha256=hashlib.sha256(blocks_file.read_bytes()).hexdigest(),
                     encoder_ledger=str(encoder_ledger), interval_seconds=interval,
                     status='Active', records=records, history=[])
        self.save(state)
        return state

    def lookup(self, record):
        code, output = self.remote(['get', 'job', '--display_name_regex', '^'+re.escape(record['name'])+'$',
                                    '--page_size', '100'])
        if code:
            raise RuntimeError(f'Cannot reconcile submission {record["name"]}: {output[-800:]}')
        for job_id in dict.fromkeys(re.findall(r'\b(dlc[a-z0-9]{10,})\b', output)):
            code, detail = self.remote(['get', 'job', job_id, '--show_detail'])
            if code:
                raise RuntimeError(f'Cannot inspect matching job: {detail[-800:]}')
            data = json.loads(detail)
            if data.get('DisplayName') == record['name']:
                if data['UserCommand'] != record['command']:
                    raise RuntimeError('Existing job name matches but command differs; refusing another submission')
                return job_id, data
        return None

    def accepted(self, record, job_id, detail=None):
        record.update(phase='Submitted', downstream_job_id=job_id, submitted_at=utc())
        record.pop('last_error', None)
        if detail:
            record['downstream_status'] = detail['Status']

    def check_record(self, state, record):
        if record['phase'] == 'Submitted':
            return
        if record.get('command'):
            # Reconcile even a crash/timeout after the request reached DLC.
            existing = self.lookup(record)
            if existing:
                self.accepted(record, *existing)
                self.log(f'{record["method"]}: reconciled {record["downstream_job_id"]}')
                self.save(state)
                return
        code, output = self.remote(['get', 'job', record['encoder_job_id'], '--show_detail'])
        if code:
            raise RuntimeError(f'Encoder status query failed: {output[-800:]}')
        detail = json.loads(output)
        record.update(encoder_status=detail['Status'], last_encoder_check_at=utc())
        if detail['Status'] != 'Succeeded':
            record['phase'] = 'EncoderFailed' if detail['Status'] in ('Failed', 'Stopped') else 'WaitingEncoder'
            self.log(f'{record["method"]}: encoder {detail["Status"]}; downstream not submitted')
            self.save(state)
            return
        if not record.get('command'):
            source, frozen = Path(record['source_checkpoint']), Path(record['frozen_checkpoint'])
            self.validate(source, record)
            freeze_checkpoint(source, frozen)
            record['encoder_metadata'] = self.validate(frozen, record)
            environment = dict(record['environment'], ENCODER_CKPT=str(frozen),
                               OUTPUT_DIR=record['output_dir'], PYTHON_BIN=PYTHON)
            record['command'] = shlex.join(['env', *[f'{k}={v}' for k, v in environment.items()],
                                          'bash', str(ROOT / 'training/train_starvla_groot.sh')])
            record['phase'] = 'Ready'
            command_file = self.folder / 'commands' / (record['method']+'.sh')
            command_file.parent.mkdir(exist_ok=True)
            command_file.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+record['command']+'\n')
            self.save(state)
            existing = self.lookup(record)
            if existing:
                self.accepted(record, *existing)
                self.save(state)
                return
        record.update(phase='Submitting', submit_attempt_at=utc())
        self.save(state)
        code, output = self.remote(['submit', 'pytorchjob', '--name='+record['name'],
                                    '--command='+record['command'], *COMMON, *RESOURCES])
        ids = re.findall(r'\b(dlc[a-z0-9]{10,})\b', output)
        record['submit_output'] = output
        if code == 0 and ids:
            self.accepted(record, ids[-1])
            self.log(f'{record["method"]}: submitted {record["downstream_job_id"]} (4 GPUs, global batch 64)')
        else:
            record['phase'] = 'SubmissionUncertain'
            # Refresh by name before deciding what the next hourly run should do.
            existing = self.lookup(record)
            if existing:
                self.accepted(record, *existing)
            else:
                record['last_error'] = f'Submit returned {code}: {output[-1000:]}'
        self.save(state)

    def tick(self):
        state = json.loads(self.path.read_text())
        state['last_check_at'] = utc()
        for record in state['records']:
            try:
                self.check_record(state, record)
            except Exception as error:
                record['last_error'] = f'{type(error).__name__}: {error}'
                self.log(f'{record["method"]}: {record["last_error"]}; retry/reconcile next check')
                self.save(state)
        counts = {phase: sum(r['phase'] == phase for r in state['records'])
                  for phase in sorted({r['phase'] for r in state['records']})}
        state['history'].append(dict(time=utc(), counts=counts))
        if all(r['phase'] == 'Submitted' for r in state['records']):
            state.update(status='Complete', completed_at=utc(), next_check_at=None)
        else:
            next_time = time.time()+state['interval_seconds']
            state['next_check_at'] = datetime.fromtimestamp(next_time, timezone.utc).isoformat()
        self.save(state)
        return state


def process_alive(pid, folder):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
        return 'watch_encoder_starvla.py' in command and str(folder) in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('start', 'run', 'once', 'status', 'stop'):
        action.add_argument('--'+name, action='store_true')
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--blocks', type=Path, default=ROOT / '2.sh')
    parser.add_argument('--encoder-ledger', type=Path, default=DEFAULT_LEDGER)
    parser.add_argument('--interval', type=int, default=3600)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be positive')
    folder = args.state_dir.resolve()
    monitor = Monitor(folder)
    if args.status:
        print(monitor.path.read_text())
        return
    if args.stop:
        pid = int((folder / 'monitor.pid').read_text())
        if process_alive(pid, folder):
            os.kill(pid, signal.SIGTERM)
            print(f'Stopped monitor PID {pid}; existing DLC jobs continue')
        return
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'setup.lock').open('a') as setup_lock:
        fcntl.flock(setup_lock, fcntl.LOCK_EX)
        monitor.initialize(args.blocks.resolve(), args.encoder_ledger.resolve(), args.interval)
    if args.start:
        state = json.loads(monitor.path.read_text())
        if state['status'] == 'Complete':
            print('All downstream jobs already submitted; monitor is complete')
            return
        pid_file = folder / 'monitor.pid'
        if pid_file.exists() and process_alive(int(pid_file.read_text()), folder):
            print(f'Monitor already running: PID {pid_file.read_text().strip()}')
            return
        with (folder / 'monitor.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--run',
                '--state-dir', str(folder), '--blocks', str(args.blocks.resolve()),
                '--encoder-ledger', str(args.encoder_ledger.resolve()), '--interval', str(args.interval)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        for _ in range(50):
            if pid_file.exists() and process_alive(int(pid_file.read_text()), folder):
                print(f'Monitor started: PID {pid_file.read_text().strip()}, interval {args.interval}s')
                print(f'State: {monitor.path}\nLog: {folder / "monitor.log"}')
                return
            if child.poll() is not None:
                raise RuntimeError('Monitor exited during startup; inspect monitor.log')
            time.sleep(.1)
        raise RuntimeError('Monitor startup not confirmed; inspect monitor.log before retrying')
    with (folder / 'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another monitor already owns this state directory')
        (folder / 'monitor.pid').write_text(str(os.getpid()))
        if args.once:
            monitor.tick()
            return
        while True:
            state = monitor.tick()
            if state['status'] == 'Complete':
                monitor.log('All five downstream jobs have been submitted; monitor exiting')
                return
            deadline = time.monotonic()+state['interval_seconds']
            while time.monotonic() < deadline:
                time.sleep(min(60, deadline-time.monotonic()))


if __name__ == '__main__':
    main()
