"""Monitor local encoder completion and submit prepared policy jobs once."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_encoder_dlc import PYTHON, remote
from scripts.watch_encoder_starvla import Monitor, freeze_checkpoint, utc


def local_status(record):
    """Only an explicit successful launcher exit makes a local run eligible."""
    if not record.get('local_encoder_run'):
        return 'Succeeded', {}
    folder = Path(record['local_encoder_run'])
    exit_file = folder / 'exit_code'
    if exit_file.exists():
        code = int(exit_file.read_text().strip())
        return ('Succeeded' if code == 0 else 'Failed'), {'exit_code': code}
    launch = json.loads((folder / 'launch.json').read_text())
    pid = launch['launcher_pid']
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return 'Failed', {'reason': 'Launcher is missing without a successful exit record'}
    if str(folder / 'launch.sh') not in command or state == 'Z':
        return 'Failed', {'reason': 'Launcher identity does not match or process is terminal'}
    details = {'pid': pid}
    metrics = folder / 'metrics.jsonl'
    if metrics.exists():
        for line in reversed(metrics.read_text().splitlines()):
            try:
                latest = json.loads(line)
                details['latest_metrics'] = latest
                break
            except json.JSONDecodeError:
                continue
    return 'Running', details


def validate_checkpoint(path, record):
    code = '''import json,sys,torch
s=torch.load(sys.argv[1],map_location="cpu",weights_only=True,mmap=True)
assert isinstance(s.get("encoder"),dict) and s["encoder"], "Missing encoder tensors"
keys=("initialization","pretrained_mode","method","size","input_mode","color_order","epoch","step","val_loss")
print(json.dumps({k:s.get(k) for k in keys}))
'''
    result = subprocess.run([PYTHON, '-c', code, str(path)], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'Checkpoint validation failed: {result.stderr[-1000:]}')
    metadata = json.loads(result.stdout)
    env = record['environment']
    for key, value in {'initialization': env['ENCODER_PRETRAIN_INIT'],
                       'pretrained_mode': env['ENCODER_PRETRAINED_MODE'],
                       'method': env['ENCODER_METHOD'], 'size': env['ENCODER_SIZE'],
                       'input_mode': env['TACTILE_INPUT_MODE'], 'color_order': 'RGB'}.items():
        if metadata.get(key) != value:
            raise ValueError(f'{key}: checkpoint={metadata.get(key)!r}, expected={value!r}')
    return metadata


class LocalPolicyMonitor(Monitor):
    def __init__(self, folder, remote_call=remote, validator=validate_checkpoint, readiness=local_status):
        self.real_remote = remote_call
        self.readiness = readiness
        super().__init__(folder, remote_call=self.route, validator=validator)

    def route(self, arguments):
        if arguments[:2] == ['get', 'job'] and arguments[2].startswith('local:'):
            state = json.loads(self.path.read_text())
            record = next(r for r in state['records'] if r['encoder_job_id'] == arguments[2])
            status, _ = self.readiness(record)
            return 0, json.dumps({'Status': status})
        return self.real_remote(arguments)

    def check_record(self, state, record):
        if record['phase'] == 'Submitted':
            return
        status, detail = self.readiness(record)
        record.update(encoder_status=status, local_progress=detail, last_encoder_check_at=utc())
        if status != 'Succeeded':
            record['phase'] = 'EncoderFailed' if status == 'Failed' else 'WaitingEncoder'
            self.log(f'{record["method"]}: local encoder {status}; downstream waiting')
            self.save(state)
            return
        if not record.get('command'):
            source = Path(record['source_checkpoint'])
            self.validate(source, record)
            frozen = Path(record['frozen_checkpoint'])
            freeze_checkpoint(source, frozen)
            record['encoder_metadata'] = self.validate(frozen, record)
            environment = dict(record['environment'], ENCODER_INIT='latest', ENCODER_CKPT=str(frozen),
                               OUTPUT_DIR=record['output_dir'], PYTHON_BIN=PYTHON)
            launcher = Path(state['source_root']) / 'training/train_starvla_groot.sh'
            record['command'] = shlex.join(['env', *[f'{k}={v}' for k, v in environment.items()], 'bash', str(launcher)])
            record['phase'] = 'Ready'
            commands = self.folder / 'commands'
            commands.mkdir(exist_ok=True)
            (commands / f'{record["method"]}.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+record['command']+'\n')
            self.save(state)
        super().check_record(state, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    folder = args.state_dir.resolve()
    monitor = LocalPolicyMonitor(folder)
    if args.status:
        print(monitor.path.read_text())
        return
    with (folder / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (folder / 'monitor.pid').write_text(str(os.getpid()))
        while True:
            state = monitor.tick()
            if args.once or state['status'] == 'Complete':
                return
            deadline = time.monotonic()+state['interval_seconds']
            while time.monotonic() < deadline:
                time.sleep(min(60, deadline-time.monotonic()))


if __name__ == '__main__':
    main()
