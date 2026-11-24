"""Contact-only interpolation calibration; TPS is a reference, not ground truth."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from encoder.tactile import dense_displacement, sensor_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/mnt/yichi_data/UniVTAC_downloads/data/contact')
    p.add_argument('--output', default='artifacts/tactile_interpolation')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    samples = []
    # One episode per object, both sensors, high displacement and high temporal
    # change. Calibration never reads the policy-derived evaluation archive.
    for folder in sorted(Path(args.root).iterdir()):
        files = sorted((folder/'hdf5').glob('*.hdf5'))
        if not files:
            continue
        with h5py.File(files[0]) as f:
            for side in ('left', 'right'):
                g = sensor_group(f, side)
                m = g['marker'][:]
                disp = m[:, 1]-m[:, 0]
                strength = np.linalg.norm(disp, axis=-1).mean(-1)
                change = np.zeros(len(m)); change[1:] = np.linalg.norm(np.diff(disp, axis=0), axis=-1).mean(-1)
                for t in sorted(set((int(strength.argmax()), int(change.argmax())))):
                    samples.append((str(files[0]), side, t, g['depth'][t], m[t]))
    methods = [('tps', 28), ('linear', 28)]+[('tps_bilinear', g) for g in (14, 28, 56)]
    rows = []; examples = []
    with threadpool_limits(limits=1):
        for i, (path, side, t, depth, markers) in enumerate(samples):
            fields = {}
            for method, grid in methods:
                key = method if method != 'tps_bilinear' else f'{method}_{grid}'
                timings = []
                for _ in range(3):
                    start = time.perf_counter()
                    field = dense_displacement(markers, depth.shape, method=method, grid=grid)
                    timings.append((time.perf_counter()-start)*1000)
                fields[key] = field
                reference = fields['tps']
                error = np.linalg.norm(field-reference, axis=-1)
                def encode(x):
                    return .5+.5*x/np.maximum(30., np.linalg.norm(x, axis=-1, keepdims=True))
                rows.append(dict(source=path, side=side, step=t, method=key,
                                 ms=float(np.median(timings)), mean_epe_px=float(error.mean()),
                                 p95_epe_px=float(np.quantile(error, .95)),
                                 max_epe_px=float(error.max()),
                                 encoded_mae=float(np.abs(encode(field)-encode(reference)).mean()),
                                 magnitude_px=float(np.linalg.norm(reference, axis=-1).max())))
            if i < 3:
                examples.append(fields)
            if (i+1)%10 == 0:
                print(f'Calibrated {i+1}/{len(samples)} frames', flush=True)
    summary = {}
    for key in sorted(set(r['method'] for r in rows)):
        group = [r for r in rows if r['method'] == key]
        summary[key] = {k: float(np.mean([r[k] for r in group])) for k in
                        ('ms', 'mean_epe_px', 'p95_epe_px', 'encoded_mae')}
        summary[key]['worst_frame_mean_epe_px'] = max(r['mean_epe_px'] for r in group)
        summary[key]['max_epe_px'] = max(r['max_epe_px'] for r in group)
    report = dict(note='Contact calibration only. Error is disagreement with dense TPS, not physical ground-truth error. Timings include deduplication/fit/evaluation, exclude I/O; one CPU thread, median of three calls.',
                  frames=len(samples), summary=summary, rows=rows)
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), len(methods), figsize=(16, 3*len(examples)), squeeze=False)
    for row, fields in enumerate(examples):
        vmax = max(np.linalg.norm(v, axis=-1).max() for v in fields.values())
        for ax, (name, field) in zip(axes[row], fields.items()):
            ax.imshow(np.linalg.norm(field, axis=-1), vmin=0, vmax=vmax, cmap='magma')
            ax.set_title(name); ax.axis('off')
    fig.suptitle('Same marker controls: displacement magnitude, shared scale per row')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=140); plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
