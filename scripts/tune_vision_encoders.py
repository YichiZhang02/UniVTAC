"""Select per-model adaptation parameters on radial data; reuse across representations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import fcntl
import os
from pathlib import Path
import subprocess
import shutil
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from encoder.pretrained_registry import SOURCES, source_dir, result_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', default=['DINOv3', 'SigLIP2', 'V-JEPA2', 'Wan2.2-VAE'])
    p.add_argument('--gpus', nargs='+', default=['0', '1'])
    p.add_argument('--samples', type=int, default=8192)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=64)
    p.add_argument('--max-val-steps', type=int, default=32)
    args = p.parse_args()
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root = ROOT / 'encoder_results' / f'vision_tuning_{stamp}'
    root.mkdir(parents=True)
    report = dict(protocol=vars(args), input_mode='depth_deform_nonlinear_radial', selection='minimum validation loss within each model',
                  search='lr={1e-5,3e-5,1e-4} at wd=.05, then wd=.01 at winning lr', models={})

    def tune(method, gpu):
        size = 'official'
        folder = source_dir('vision', method)
        if not list(folder.rglob('*.safetensors')):
            return method, dict(status='blocked', reason=f'Official weights missing: {folder}')
        batch = 64
        trials = []

        def trial(lr, wd):
            trial_id = f'pilot_{stamp}_lr{lr:g}_wd{wd:g}'
            output = result_root('depth_deform_nonlinear_radial', 'vision', method, size) / trial_id
            cmd = [sys.executable, '-u', str(ROOT / 'encoder/train_external.py'), '--pretrained-mode', 'vision',
                   '--method', method, '--size', size, '--init', 'adapted', '--input-mode', 'depth_deform_nonlinear_radial',
                   '--output-dir', str(output), '--lr', str(lr), '--weight-decay', str(wd),
                   '--batch-size', str(batch), '--samples', str(args.samples), '--epochs', str(args.epochs),
                   '--max-steps', str(args.max_steps), '--max-val-steps', str(args.max_val_steps),
                   '--workers', '4', '--no-training-state']
            if ',' in gpu:
                cmd.append('--multi-gpu')
            logfile = root / f'{method}_{trial_id}.log'
            with logfile.open('w') as stream:
                done = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                      stdout=stream, stderr=subprocess.STDOUT)
            record = dict(lr=lr, weight_decay=wd, batch_size=batch, gpu_group=gpu,
                          output=str(output), log=str(logfile), exit_code=done.returncode)
            if done.returncode == 0:
                metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
                record['val_loss'] = min(x['val_loss'] for x in metrics if 'val_loss' in x)
            trials.append(record)
            print(method, json.dumps(record), flush=True)
            (root / f'{method}.json').write_text(json.dumps(trials, indent=2))
            return record

        for lr in [1e-5, 3e-5, 1e-4]:
            trial(lr, .05)
        valid = [x for x in trials if 'val_loss' in x]
        if not valid:
            return method, dict(status='failed', trials=trials)
        best = min(valid, key=lambda x: x['val_loss'])
        trial(best['lr'], .01)
        best = min([x for x in trials if 'val_loss' in x], key=lambda x: x['val_loss'])
        selected = result_root('depth_deform_nonlinear_radial', 'vision', method, size) / f'{stamp}_selected'
        selected.mkdir(exist_ok=True)
        shutil.copy2(Path(best['output']) / 'encoder.pth', selected / '.encoder.pth.tmp')
        os.replace(selected / '.encoder.pth.tmp', selected / 'encoder.pth')
        # Discovery sorts by mtime, so the selected initializer must be newer than every pilot.
        os.utime(selected / 'encoder.pth', None)
        cfg = json.loads((Path(best['output']) / 'config.json').read_text())
        cfg.update(selected_from=best['output'], output_dir=str(selected))
        (selected / 'config.json').write_text(json.dumps(cfg, indent=2))
        best['selected_checkpoint'] = str(selected / 'encoder.pth')
        return method, dict(status='selected' if all(x['exit_code'] == 0 for x in trials) else 'partial', best=best, trials=trials)

    # Finish other queues before reserving the full GPU set for Wan.
    def queue(methods, gpu):
        return [tune(method, gpu) for method in methods]

    devices = list(dict.fromkeys(device for group in args.gpus for device in group.split(',')))
    shared_wan = 'Wan2.2-VAE' in args.methods and len(devices) > 1
    queued_methods = [method for method in args.methods if method != 'Wan2.2-VAE' or not shared_wan]
    with ThreadPoolExecutor(len(args.gpus)) as pool:
        jobs = [pool.submit(queue, queued_methods[i::len(args.gpus)], gpu) for i, gpu in enumerate(args.gpus)]
        for job in as_completed(jobs):
            for method, result in job.result():
                report['models'][method] = result
            (root / 'report.json').write_text(json.dumps(report, indent=2))
    if shared_wan:
        method, result = tune('Wan2.2-VAE', ','.join(devices))
        report['models'][method] = result
        (root / 'report.json').write_text(json.dumps(report, indent=2))
    selected = {}
    for method, result in report['models'].items():
        if result['status'] != 'selected':
            continue
        best = result['best']
        selected[method] = {k: best[k] for k in ['lr', 'weight_decay', 'batch_size']}
        selected[method]['epochs'] = 5
        if ',' in best.get('gpu_group', ''):
            selected[method]['multi_gpu'] = True
    config_path = ROOT / 'encoder/vision_config.yml'
    with (ROOT / 'encoder_results/.vision_config.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = (yaml.safe_load(config_path.read_text()) or {}) if config_path.exists() else {}
        previous.update(selected)
        temporary = config_path.with_suffix('.yml.tmp')
        temporary.write_text('# Shared across representations; completed radial searches update model entries.\n' + yaml.safe_dump(previous, sort_keys=False))
        os.replace(temporary, config_path)
        from encoder.catalog import canonical_name
        for method, settings in selected.items():
            model_config = ROOT / 'encoder' / canonical_name('vision', method) / 'config.yml'
            model_config.write_text(yaml.safe_dump(settings, sort_keys=False))
    print(f'Report: {root / "report.json"}', flush=True)
    if len(selected) != len(args.methods):
        sys.exit(1)


if __name__ == '__main__':
    main()
