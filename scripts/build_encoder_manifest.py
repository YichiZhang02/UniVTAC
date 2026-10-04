"""Cache tactile HDF5 frame counts for repeated encoder training runs."""
import argparse
import json
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    parser.add_argument('--output', type=Path, default=ROOT / 'encoder_results/.contact_lengths.json')
    args = parser.parse_args()
    files = sorted(args.data_root.glob('*/hdf5/*.hdf5'))
    lengths = {}
    for path in files:
        with h5py.File(path, 'r') as h5:
            lengths[str(path)] = {}
            for side in ('left', 'right'):
                group = h5[f'tactile/{side}_tactile']
                lengths[str(path)][side] = {
                    kind: len(group[kind]) for kind in ('rgb', 'rgb_marker', 'marker', 'depth')
                    if kind in group}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'files': [str(path) for path in files],
                                       'lengths': lengths}))
    print(f'Indexed {len(files)} files: {args.output}')


if __name__ == '__main__':
    main()
