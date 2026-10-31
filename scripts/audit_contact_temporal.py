"""Audit chronological four-frame contact windows without loading entire image arrays."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'encoder')]
from encoder.pretrained_registry import history_indices


def audit(data_root):
    files = sorted(data_root.glob('*/hdf5/*.hdf5'))
    if not files:
        raise ValueError(f'No contact trajectories in {data_root}')
    lengths, differences, examples, sampled_classes = [], Counter(), [], set()
    for path in files:
        with h5py.File(path, 'r') as h5:
            steps = h5['step'][:].reshape(-1)
            delta = np.diff(steps)
            if len(delta) and (delta[0] <= 0 or not np.all(delta == delta[0])):
                raise AssertionError(f'Noncontinuous steps: {path}')
            differences.update(map(int, delta))
            lengths.append(len(steps))
            for side in ('left', 'right'):
                for kind in ('depth', 'marker', 'rgb_marker'):
                    assert len(h5[f'tactile/{side}_tactile/{kind}']) == len(steps), (path, side, kind)
            shape = path.parent.parent.name
            if shape in sampled_classes or len(steps) < 4:
                continue
            sampled_classes.add(shape)
            anchor = max(3, len(steps) // 2)
            indices = history_indices(anchor, 4, 1)
            assert len(set(indices)) == 4
            # Adjacent saved observations differ by the required two simulation steps.
            assert np.all(np.diff(steps[indices]) == 2)
            for side in ('left', 'right'):
                depth = h5[f'tactile/{side}_tactile/depth'][indices]
                cropped = depth[:, 50:-50, 50:-50]
                contact = np.ptp(cropped, axis=(1, 2)) > 1e-6
                assert contact.all(), (path, side, indices, 'no contact variation')
                example = dict(file=str(path), side=side, indices=indices,
                               steps=steps[indices].tolist(), all_contact=bool(contact.all()),
                               mean_depth_frame_change=float(np.abs(np.diff(depth, axis=0)).mean()))
                examples.append(example)
    return dict(files=len(files), object_classes=len(sampled_classes),
                timepoints=sum(lengths), min_length=min(lengths), max_length=max(lengths),
                median_length=float(np.median(lengths)), step_differences=dict(differences),
                complete_4frame_simulation_stride2_windows_both_sides=sum(max(0, n-3) for n in lengths)*2,
                vjepa_windows_with_future_horizon1=sum(max(0, n-4) for n in lengths)*2,
                checked_contact_windows=len(examples), examples=examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/temporal_validation/contact_audit.json')
    args = parser.parse_args()
    report = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'examples'}, indent=2))
