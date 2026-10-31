"""Hourly per-encoder completion -> local StarVLA validation -> deduplicated DLC."""
import argparse
from contextlib import contextmanager
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from encoder.catalog import MULTIFRAME_METHODS
from scripts.submit_encoder_dlc import PYTHON, remote
from scripts.watch_encoder_starvla import atomic_json, freeze_checkpoint, utc
from scripts.watch_policy_dlc import LocalPolicyMonitor


def parse_blocks(path):
    blocks, environment = {}, None
    for line in Path(path).read_text().splitlines():
        line = line.strip().rstrip(' \\')
        if line.startswith('CUDA_VISIBLE_DEVICES='):
            environment = {}
        if environment is None:
            continue
        if re.match(r'^[A-Z_]+=', line):
            tokens = shlex.split(line)
            if len(tokens) != 1:
                raise ValueError(f'Invalid environment assignment: {line}')
            key, value = tokens[0].split('=', 1)
            environment[key] = value
        elif line.startswith('bash '):
            if shlex.split(line) != ['bash', str(ROOT / 'training/train_starvla_groot.sh')]:
                raise ValueError('Expected the StarVLA-GR00T launcher')
            method = environment['ENCODER_METHOD']
            if method in blocks or method not in MULTIFRAME_METHODS:
                raise ValueError(f'Duplicate or unsupported encoder: {method}')
            expected = dict(NUM_PROCESSES='4', CUDA_VISIBLE_DEVICES='0,1,2,3',
                            TACTILE_TYPE='pool3x3', TACTILE_MODE='encode',
                            TACTILE_INSERT_LOCATION='encoder', ENCODER_PRETRAINED_MODE='univtac',
                            ENCODER_PRETRAIN_INIT='scratch', ENCODER_INIT='latest', ENCODER_SIZE='S',
                            TACTILE_INPUT_MODE='depth_deform_nonlinear_radial')
            if any(environment.get(k) != v for k, v in expected.items()):
                raise ValueError(f'{method}: unexpected policy configuration')
            blocks[method] = environment
            environment = None
    if set(blocks) != set(MULTIFRAME_METHODS):
        raise ValueError('Expected exactly eight temporal encoder blocks')
    return blocks


def temporal_status(record):
    state = json.loads(Path(record['encoder_status_file']).read_text())
    job = next(job for job in state['jobs'] if job['method'] == record['method'])
    if (Path(job['output_dir']) / 'encoder.pth').resolve() != Path(record['source_checkpoint']).resolve():
        raise ValueError('Training output no longer matches the pinned checkpoint')
    if state.get('status') == 'Reconfigured':
        return 'Failed', {'reason': 'Encoder queue was superseded'}
    if job['state'] == 'complete' and job.get('returncode') == 0:
        return 'Succeeded', job
    if job['state'] in ('failed', 'complete'):
        return 'Failed', job
    return 'Running', job


def validate_checkpoint(path, record):
    code = '''import json,math,sys,torch
s=torch.load(sys.argv[1],map_location='cpu',weights_only=True,mmap=True)
assert s.get('encoder') and math.isfinite(s['val_loss'])
keys=('method','size','input_mode','initialization','pretrained_mode','color_order','num_frames',
      'frame_stride','simulation_step_stride','temporal_backend','temporal_compression','output_frames',
      'sd_patch_size','epoch','step','val_loss')
print(json.dumps({k:s.get(k) for k in keys}))
'''
    result = subprocess.run([PYTHON, '-c', code, str(path)], text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    metadata = json.loads(result.stdout)
    expected = dict(method=record['method'], size='S', input_mode='depth_deform_nonlinear_radial',
                    initialization='scratch', pretrained_mode='univtac', color_order='RGB',
                    num_frames=4, frame_stride=1, simulation_step_stride=2,
                    temporal_backend='conv3d', temporal_compression='4in2', output_frames=2)
    if record['method'] == 'SD-VAE4Frame':
        expected['sd_patch_size'] = 2
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'{key}={metadata.get(key)!r}, expected {value!r}')
    if not 1 <= metadata['epoch'] <= 20 or metadata['step'] < 1:
        raise ValueError('Invalid trained checkpoint progress')
    return metadata


def process_key(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def watch_lease(path):
    """Resume only our paused scheduler if validation dies or exceeds its timeout."""
    while path.exists():
        lease = json.loads(path.read_text())
        if (process_key(lease['monitor_pid']) != lease['monitor_key'] or time.time() > lease['expires']):
            child = lease.get('validation_pid')
            if child and process_key(child) == lease.get('validation_key'):
                os.killpg(child, signal.SIGTERM)
                for _ in range(20):
                    if process_key(child) is None:
                        break
                    time.sleep(.25)
                if process_key(child) == lease['validation_key']:
                    os.killpg(child, signal.SIGKILL)
            if process_key(lease['scheduler_pid']) == lease['scheduler_key']:
                os.kill(lease['scheduler_pid'], signal.SIGCONT)
            return
        time.sleep(5)


@contextmanager
def validation_gpu(folder, status_file):
    """Serialize local policy checks across both hourly monitors."""
    lock_path = ROOT / '.cache/temporal_validation/local_policy_validation.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with _validation_gpu_unlocked(folder, status_file) as gpu:
            yield gpu


@contextmanager
def _validation_gpu_unlocked(folder, status_file):
    """Existing encoder children keep training; defer new launches during the smoke."""
    state = json.loads(Path(status_file).read_text())
    scheduler = state['pid']
    key = process_key(scheduler)
    lease_path = Path(folder) / 'gpu_lease.json'
    paused = False
    try:
        if key is not None:
            command = Path(f'/proc/{scheduler}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'train_multiframe_encoders.py' not in command or str(Path(status_file).parent) not in command:
                raise RuntimeError('Encoder scheduler process identity changed')
            proc_state = Path(f'/proc/{scheduler}/stat').read_text().rsplit(')', 1)[1].split()[0]
            if proc_state == 'T':
                raise RuntimeError('Encoder scheduler is already paused by another owner')
            atomic_json(lease_path, dict(scheduler_pid=scheduler, scheduler_key=key,
                monitor_pid=os.getpid(), monitor_key=process_key(os.getpid()), expires=time.time()+660))
            # Stop the scheduler, never its training children. The watchdog releases
            # the scheduler even if this monitor is killed without running finally.
            subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--watch-lease', str(lease_path)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            os.kill(scheduler, signal.SIGSTOP)
            paused = True
            time.sleep(1)
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                         '--format=csv,noheader,nounits'], text=True)
        choices = [(int(free.strip()), index.strip()) for index, free in
                   (line.split(',') for line in usage.splitlines())]
        # Leave headroom for both a per-GPU-16 policy and the existing small encoder.
        free, gpu = max(choices)
        yield gpu if free >= 65 * 1024 else None
    finally:
        if paused and process_key(scheduler) == key:
            os.kill(scheduler, signal.SIGCONT)
        lease_path.unlink(missing_ok=True)


def run_local_validation(folder, state, record):
    attempt = Path(folder) / 'local_validation' / record['method'] / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    scheduler_status = record.get('gpu_scheduler_status_file') or record['encoder_status_file']
    with validation_gpu(folder, scheduler_status) as gpu:
        if gpu is None:
            return None
        attempt.mkdir(parents=True)
        command = [PYTHON, '-u', str(Path(state['source_root']) / 'scripts/local_temporal_starvla_smoke.py'),
                   '--source', state['source_root'], '--checkpoint', record['frozen_checkpoint'],
                   '--method', record['method'], '--output', str(attempt),
                   '--reference-run', state['reference_run']]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            environment.pop(name, None)
        with (attempt / 'launcher.log').open('w') as log:
            process = subprocess.Popen(command, cwd=state['source_root'], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            lease_path = Path(folder) / 'gpu_lease.json'
            if lease_path.exists():
                lease = json.loads(lease_path.read_text())
                lease.update(validation_pid=process.pid, validation_key=process_key(process.pid))
                atomic_json(lease_path, lease)
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise RuntimeError(f'Local validation timed out: {attempt}')
        record['local_validation_dir'] = str(attempt)
        if code:
            raise RuntimeError(f'Local training failed (exit {code}): {attempt}/launcher.log')
        report = json.loads((attempt / 'validation_passed.json').read_text())
        if not (report.get('passed') and report['method'] == record['method'] and
                report.get('encoder_gradient_finite', report.get('conv3d_gradient_finite')) and
                report.get('encoder_gradient_nonzero', report.get('conv3d_gradient_nonzero'))):
            raise RuntimeError(f'Invalid local validation report: {attempt}')
        return report


class TemporalPolicyMonitor(LocalPolicyMonitor):
    def __init__(self, folder, remote_call=remote, validator=validate_checkpoint,
                 readiness=temporal_status, local_validator=run_local_validation):
        self.local_validator = local_validator
        super().__init__(folder, remote_call=remote_call, validator=validator, readiness=readiness)

    def route(self, arguments):
        if arguments[0] == 'submit':
            prefixes = ('--worker_cpu=', '--worker_memory=', '--worker_shared_memory=')
            arguments = [arg for arg in arguments if not arg.startswith(prefixes)] + [
                '--worker_cpu=32', '--worker_memory=400Gi', '--worker_shared_memory=400Gi']
            resource_id = json.loads(self.path.read_text()).get('resource_id')
            if resource_id:
                arguments = [arg for arg in arguments if not arg.startswith('--resource_id=')]
                arguments.append('--resource_id='+resource_id)
        return super().route(arguments)

    def check_record(self, state, record):
        if record['phase'] == 'Submitted':
            return
        if record.get('command'):
            existing = self.lookup(record)
            if existing:
                self.accepted(record, *existing)
                self.save(state)
                return
        status, detail = self.readiness(record)
        record.update(encoder_status=status, local_progress=detail, last_encoder_check_at=utc())
        if status != 'Succeeded':
            record['phase'] = 'EncoderFailed' if status == 'Failed' else 'WaitingEncoder'
            self.save(state)
            return
        if not record.get('local_validation', {}).get('passed'):
            self.validate(Path(record['source_checkpoint']), record)
            freeze_checkpoint(Path(record['source_checkpoint']), Path(record['frozen_checkpoint']))
            record['encoder_metadata'] = self.validate(Path(record['frozen_checkpoint']), record)
            record['phase'] = 'ValidatingLocally'
            self.save(state)
            try:
                report = self.local_validator(self.folder, state, record)
            except Exception:
                record['phase'] = 'LocalValidationFailed'
                self.save(state)
                raise
            if report is None:
                record['phase'] = 'WaitingGPU'
                self.save(state)
                return
            record.update(local_validation=report, local_validation_passed_at=utc(), phase='Ready')
            self.save(state)
        super().check_record(state, record)


def initialize(folder, blocks_file, status_file, interval):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'state.json'
    if path.exists():
        return json.loads(path.read_text())
    blocks = parse_blocks(blocks_file)
    encoders = json.loads(status_file.read_text())
    jobs = {job['method']: job for job in encoders['jobs']}
    for method, environment in blocks.items():
        if Path(environment['ENCODER_CKPT']).resolve() != (Path(jobs[method]['output_dir']) / 'encoder.pth').resolve():
            raise ValueError(f'{method}: 2.sh checkpoint does not match the encoder run')
    source = folder / 'source'
    launch = json.loads((status_file.parent / 'launch.json').read_text())
    ignore = shutil.ignore_patterns('__pycache__', '*.ipynb', '.git')
    shutil.copytree(Path(launch['source_root']) / 'encoder', source / 'encoder', ignore=ignore)
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
    shutil.copy2(blocks_file, folder / '2.sh')
    reference = folder / 'reference'
    reference.mkdir()
    for name in ('train_config.yml', 'episodes.json', 'dataset_stats.json'):
        shutil.copy2(ROOT / 'policy_results/dlc_1sh_20261028_124426/runs/SD-VAE' / name, reference / name)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    records = []
    for method, environment in blocks.items():
        checkpoint = Path(environment['ENCODER_CKPT']).resolve()
        if checkpoint != (Path(jobs[method]['output_dir']) / 'encoder.pth').resolve():
            raise ValueError(f'{method}: 2.sh checkpoint does not match the encoder run')
        environment.update(WORKERS='2', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                           NCCL_DEBUG='WARN', NCCL_SOCKET_IFNAME='eth0')
        records.append(dict(method=method, environment=environment, phase='WaitingEncoder',
            name=f'yichi_temporal_{stamp}_{method.lower().replace("-", "")}_pool3x3',
            encoder_job_id='local:'+method, encoder_status_file=str(status_file),
            source_checkpoint=str(checkpoint), frozen_checkpoint=str(folder / 'checkpoints' / method / 'encoder.pth'),
            output_dir=str(folder / 'runs' / method)))
    state = dict(status='Active', created_at=utc(), interval_seconds=interval, records=records,
                 source_root=str(source), encoder_status_file=str(status_file), history=[],
                 reference_run=str(reference),
                 source_blocks=str(blocks_file), source_blocks_sha256=hashlib.sha256(blocks_file.read_bytes()).hexdigest())
    atomic_json(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('start', 'run', 'once', 'status', 'stop'):
        action.add_argument('--'+name, action='store_true')
    action.add_argument('--ready', metavar='METHOD')
    action.add_argument('--watch-lease', type=Path)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--blocks', type=Path, default=ROOT / '2.sh')
    parser.add_argument('--encoder-status', type=Path)
    parser.add_argument('--interval', type=int, default=3600)
    args = parser.parse_args()
    if args.watch_lease:
        watch_lease(args.watch_lease)
        return
    if args.ready:
        state = json.loads(args.encoder_status.read_text())
        job = next(job for job in state['jobs'] if job['method'] == args.ready)
        if (state.get('status') != 'Reconfigured' and job['state'] == 'complete' and
                job.get('returncode') == 0 and (Path(job['output_dir']) / 'encoder.pth').is_file()):
            return
        print(f'Skipping {args.ready}: encoder training is {job["state"]}', file=sys.stderr)
        raise SystemExit(1)
    if args.state_dir is None or args.interval < 1:
        parser.error('--state-dir and a positive interval are required')
    folder = args.state_dir.resolve()
    if args.status:
        print((folder / 'state.json').read_text())
        return
    if args.stop:
        pid = int((folder / 'monitor.pid').read_text())
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        if 'watch_temporal_starvla.py' in command and str(folder) in command:
            os.kill(pid, signal.SIGTERM)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        initialize(folder, args.blocks.resolve(), args.encoder_status.resolve(), args.interval)
    if args.start:
        pid_file = folder / 'monitor.pid'
        if pid_file.exists() and process_key(int(pid_file.read_text())):
            raise RuntimeError('A monitor already exists for this state directory')
        with (folder / 'monitor.log').open('a') as log:
            process = subprocess.Popen([PYTHON, '-u', str(Path(__file__).resolve()), '--run',
                '--state-dir', str(folder), '--encoder-status', str(args.encoder_status.resolve()),
                '--blocks', str(args.blocks.resolve()), '--interval', str(args.interval)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Monitor PID {process.pid}; state {folder / "state.json"}', flush=True)
        return
    with (folder / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (folder / 'monitor.pid').write_text(str(os.getpid()))
        monitor = TemporalPolicyMonitor(folder)
        while True:
            state = monitor.tick()
            monitor.log(str({r['method']: r['phase'] for r in state['records']}))
            if args.once or state['status'] == 'Complete':
                return
            deadline = time.monotonic() + state['interval_seconds']
            while time.monotonic() < deadline:
                time.sleep(min(60, deadline-time.monotonic()))


if __name__ == '__main__':
    main()
