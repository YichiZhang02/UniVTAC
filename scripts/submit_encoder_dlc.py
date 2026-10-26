"""Submit Sparsh/Wan pretraining to the existing UniVTAC DLC workspace.

Run --submit to freeze sources and submit scratch jobs (Sparsh 1 GPU, Wan 4).
Use --ledger <path> --status to inspect the same jobs without resubmitting.
Credentials are refreshed for every request and never written to disk.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/mnt/yichi_data/miniconda3/envs/univtac/bin/python'
MODELS = {'MAE': ('tactile', 'S'), 'DINOv2': ('tactile', 'B'),
          'I-JEPA': ('tactile', 'S'), 'V-JEPA': ('tactile', 'S'),
          'Wan2.2-VAE': ('vision', 'B')}
COMMON = [
    '--data_sources=d-lu3fmaylj6r0n6hkms:v1:/mnt/yichi_data/',
    '--resource_id=quota3ya3nugiuc6', '--vpc_id=vpc-0jlibv5lchq2ljpnl6p4i',
    '--switch_id=vsw-0jlkzeskp0wfxqbgged89', '--security_group_id=sg-0jl5nh21jy9r5uts3pk6',
    '--priority=1', '--default_route=eth0', '--driver=580.95.05-open',
    '--extended_cidrs=10.11.65.0/24,10.11.64.0/24', '--success_policy=All', '--workers=1',
    '--worker_image=daimon-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/daimon/zhangyichi:univtac-train-1001',
]


def remote(arguments):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:7002/api/v1/credentials/0', timeout=15) as response:
        credentials = json.load(response)
    keys = ('AccessKeyId', 'AccessKeySecret', 'SecurityToken')
    if not all(credentials.get(key) for key in keys):
        raise RuntimeError('Incomplete STS credentials')
    auth = ['--region=cn-wulanchabu', '--endpoint=pai-dlc.cn-wulanchabu.aliyuncs.com',
            '-a', credentials[keys[0]], '-k', credentials[keys[1]], '-S', credentials[keys[2]]]
    if arguments[0] not in ('logs', 'stop') and arguments[:2] != ['get', 'pod']:
        auth += ['--workspace_id=218916']
    try:
        result = subprocess.run(['dlc', *arguments, *auth], text=True, capture_output=True, timeout=55)
    except subprocess.TimeoutExpired:
        return -1, 'Timed out; query the unique job name before attempting another submission.'
    output = result.stdout + result.stderr
    for key in keys:
        output = output.replace(credentials[key], '[REDACTED]')
    return result.returncode, output


def save(path, records):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    temporary.replace(path)


def freeze_checkpoint(source, destination):
    """Published checkpoints use atomic replacement, so hardlinks freeze an inode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(args, ledger):
    if ledger.exists():
        return json.loads(ledger.read_text())
    source = ledger.parent / 'source'
    source.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / 'encoder', source / 'encoder', ignore=shutil.ignore_patterns('__pycache__', '*.ipynb'))
    shutil.copy2(ROOT / 'train_encoder.sh', source / 'train_encoder.sh')
    (source / 'resources').symlink_to(ROOT / 'resources', target_is_directory=True)
    (source / 'encoder_results').symlink_to(ROOT / 'encoder_results', target_is_directory=True)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((source / 'encoder').rglob('*')) if p.is_file()}
    hashes['train_encoder.sh'] = hashlib.sha256((source / 'train_encoder.sh').read_bytes()).hexdigest()
    (ledger.parent / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    records = []
    previous = {}
    for path in ([args.resume_ledger] if args.resume_ledger else []) + args.additional_resume_ledger:
        previous.update({r['method']: r for r in json.loads(path.read_text())})
    for method in args.methods:
        mode, size = MODELS[method]
        gpus = args.wan_gpus if method == 'Wan2.2-VAE' else args.gpus
        key = method.lower().replace('.', '').replace('-', '')
        output = ROOT / 'encoder_results' / args.input / mode / method / size / args.init / f'{stamp}_dlc_{args.init}'
        resume_args, resume_info = [], {}
        if previous:
            old = previous[method]
            old_output = Path(old['output_dir'])
            original = old_output / 'last.pth'
            if not original.is_file() and old.get('resume_checkpoint'):
                original = Path(old['resume_checkpoint'])
            if not original.is_file():
                raise FileNotFoundError(f'{method}: no completed epoch checkpoint: {original}')
            frozen = ledger.parent / 'resume_snapshots' / method / 'last.pth'
            freeze_checkpoint(original, frozen)
            probe = subprocess.run([PYTHON, '-c',
                'import torch,json,sys; s=torch.load(sys.argv[1],map_location="cpu",weights_only=True,mmap=True); '
                'print(json.dumps({"epoch":s["epoch"],"step":s["step"],"config":s["training_config"]}))',
                str(frozen)], check=True, capture_output=True, text=True)
            saved = json.loads(probe.stdout)
            config = saved['config']
            if (config.get('world_size', 1) != gpus or config['input_mode'] != args.input
                    or config.get('init', 'official') != args.init or config['samples'] != args.samples):
                raise ValueError(f'{method}: resume requires matching GPUs, representation, initialization and sample budget')
            total_epochs = args.epochs if args.epochs is not None else config['epochs']
            patience = args.early_stop_patience if args.early_stop_patience is not None else config['early_stop_patience']
            if saved['epoch'] >= total_epochs:
                raise ValueError(f'{method}: checkpoint already completed {saved["epoch"]} epochs; budget is {total_epochs}')
            resume_args = ['--resume', str(frozen), '--epochs', str(total_epochs),
                           '--lr', str(config['lr']), '--weight-decay', str(config['weight_decay']),
                           '--mask-ratio', str(config['mask_ratio']), '--kl-weight', str(config['kl_weight']),
                           '--calibration-steps', str(config['calibration_steps']),
                           '--early-stop-patience', str(patience)]
            output.mkdir(parents=True, exist_ok=True)
            for name in ('encoder.pth', 'best.pth'):
                if (old_output / name).exists():
                    freeze_checkpoint(old_output / name, output / name)
            resume_info = dict(resumed_from_job=old['job_id'], resumed_epoch=saved['epoch'],
                               resumed_step=saved['step'], resume_checkpoint=str(frozen),
                               epochs=total_epochs, early_stop_patience=patience)
        else:
            if args.epochs is not None:
                resume_args += ['--epochs', str(args.epochs)]
            if args.early_stop_patience is not None:
                resume_args += ['--early-stop-patience', str(args.early_stop_patience)]
        command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES='+','.join(map(str, range(gpus))), f'NUM_PROCESSES={gpus}',
            'OMP_NUM_THREADS=4', 'NCCL_DEBUG=WARN', 'NCCL_SOCKET_IFNAME=eth0',
            'TOKENIZERS_PARALLELISM=false', 'PYTHONUNBUFFERED=1', f'PYTHON_BIN={PYTHON}',
            'bash', str(source / 'train_encoder.sh'), '--pretrained-mode', mode,
            '--method', method, '--size', size, '--init', args.init, '--input', args.input,
            '--tactile-preprocessing', args.preprocessing,
            '--data-root', str(args.data_root), '--', '--samples', str(args.samples),
            '--workers', '2', '--seed', '42', '--early-stop-patience', '2', '--output-dir', str(output), *resume_args])
        records.append(dict(method=method, size=size, initialization=args.init,
            name=f'yichi_enc_{stamp}_{key}_{args.init}', command=command,
            output_dir=str(output), status='Prepared', source=str(source), gpus=gpus,
            batch_size=64, per_gpu_batch_size=64//gpus, tactile_preprocessing=args.preprocessing, **resume_info))
    save(ledger, records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--resume-ledger', type=Path, help='Snapshot last.pth for selected models from an earlier ledger')
    parser.add_argument('--additional-resume-ledger', type=Path, action='append', default=[])
    parser.add_argument('--epochs', type=int, help='Total epoch limit, including completed epochs when resuming')
    parser.add_argument('--early-stop-patience', type=int, help='Unimproved validation epochs before early stopping')
    parser.add_argument('--preprocessing', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--methods', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--init', choices=['scratch', 'official'], default='scratch')
    parser.add_argument('--input', default='depth_deform_nonlinear_radial')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--gpus', type=int, choices=[1, 2, 4, 8], default=1,
                        help='GPUs for each Sparsh model; global batch remains 64')
    parser.add_argument('--wan-gpus', type=int, choices=[1, 2, 4, 8], default=4)
    args = parser.parse_args()
    if args.status and not args.ledger:
        parser.error('--status requires --ledger')
    if args.ledger is None:
        args.ledger = ROOT / 'encoder_results' / ('dlc_pretraining_'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')) / 'submissions.json'
    args.ledger = args.ledger.resolve()
    args.data_root = args.data_root.resolve()
    if args.resume_ledger:
        args.resume_ledger = args.resume_ledger.resolve()
    args.additional_resume_ledger = [path.resolve() for path in args.additional_resume_ledger]
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.early_stop_patience is not None and args.early_stop_patience < 0:
        parser.error('--early-stop-patience must be nonnegative')
    if args.status:
        records = json.loads(args.ledger.read_text())
    else:
        if len(list(args.data_root.glob('*/hdf5/*.hdf5'))) < 2:
            parser.error('data root must contain contact trajectories in */hdf5/*.hdf5')
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        records = prepare(args, args.ledger)
    if args.submit:
        for record in records:
            if record['status'] != 'Prepared':
                continue
            # Persist intent before network I/O; uncertain submissions are never
            # automatically retried, avoiding duplicate paid jobs.
            record['status'] = 'Submitting'
            save(args.ledger, records)
            gpus = record['gpus']
            resources = [f'--worker_gpu={gpus}', f'--worker_cpu={8 if gpus == 1 else 16}',
                         f'--worker_memory={"64Gi" if gpus == 1 else "128Gi"}',
                         f'--worker_shared_memory={"16Gi" if gpus == 1 else "32Gi"}']
            code, output = remote(['submit', 'pytorchjob', '--name='+record['name'],
                                   '--command='+record['command'], *COMMON, *resources])
            ids = re.findall(r'\b(dlc[a-z0-9]{10,})\b', output)
            record.update(returncode=code, submit_output=output,
                          status='Submitted' if code == 0 and ids else 'Submission failed or uncertain')
            if code == 0 and ids:
                record['job_id'] = ids[-1]
            save(args.ledger, records)
            print(record['method'], record.get('job_id', ''), record['status'], flush=True)
            if record['status'] != 'Submitted':
                raise RuntimeError(output[-1500:])
    if args.submit or args.status:
        for record in records:
            if 'job_id' not in record:
                continue
            code, output = remote(['get', 'job', record['job_id'], '--show_detail'])
            if code:
                raise RuntimeError(output[-1500:])
            detail = json.loads(output)
            assert detail['UserCommand'] == record['command']
            assert detail['JobSpecs'][0]['ResourceConfig']['GPU'] == str(record.get('gpus', 4))
            record.update(status=detail['Status'], reason=detail.get('ReasonMessage', ''), detail=detail)
            save(args.ledger, records)
            print(record['method'], record['job_id'], record['status'], record['reason'], flush=True)
    else:
        print(json.dumps([{k: r[k] for k in ('method', 'initialization', 'output_dir', 'command')} for r in records], indent=2))
    print('Ledger:', args.ledger, flush=True)


if __name__ == '__main__':
    main()
