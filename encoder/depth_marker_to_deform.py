"""Convert tactile depth and fixed/current marker coordinates into depth_deform.

R is normalized depth relative to the 34 mm no-contact camera-depth baseline:
zero at rest and increasing as camera depth decreases, clipped over 10 mm.
G and B are normalized horizontal x/y displacement fields.
Marker displacement is linearly interpolated inside the marker grid and extended
to the image boundary using inverse-square-distance weighting of the eight
nearest markers (or all markers when fewer than eight are available).
"""
import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial import Delaunay, cKDTree


@lru_cache(maxsize=8)
def _interpolation_map(fixed_bytes, count, source_h, source_w, image_size):
    """Cache barycentric weights for a sensor's fixed marker grid."""
    fixed = np.frombuffer(fixed_bytes, dtype=np.float32).reshape(count, 2)
    xs = (np.arange(image_size, dtype=np.float32) + .5) * source_w / image_size
    ys = (np.arange(image_size, dtype=np.float32) + .5) * source_h / image_size
    xx, yy = np.meshgrid(xs, ys)
    query = np.column_stack((xx.ravel(), yy.ravel()))
    triangulation = Delaunay(fixed)
    simplex = triangulation.find_simplex(query)
    inside = simplex >= 0
    neighbor_count = min(8, count)
    width = max(3, neighbor_count)
    vertices = np.zeros((len(query), width), dtype=np.int32)
    weights = np.zeros((len(query), width), dtype=np.float32)
    vertices[inside, :3] = triangulation.simplices[simplex[inside]]
    affine = triangulation.transform[simplex[inside]]
    xy = np.einsum('nij,nj->ni', affine[:, :2, :], query[inside] - affine[:, 2, :])
    weights[inside, :2] = xy
    weights[inside, 2] = 1 - xy.sum(axis=1)
    if (~inside).any():
        distances, neighbors = cKDTree(fixed).query(
            query[~inside], k=neighbor_count)
        inverse_distances = 1.0 / np.maximum(distances, 1e-6) ** 2
        vertices[~inside, :neighbor_count] = neighbors
        weights[~inside, :neighbor_count] = (
            inverse_distances / inverse_distances.sum(axis=1, keepdims=True))
    return vertices, weights


def convert(depth, markers, image_size=224):
    """Return float32 CHW in [0, 1] from HxW depth and [2, N, 2] markers."""
    depth = np.asarray(depth, dtype=np.float32)
    markers = np.asarray(markers, dtype=np.float32)
    if depth.ndim != 2 or markers.ndim != 3 or markers.shape[0] != 2 or markers.shape[-1] != 2:
        raise ValueError('Expected depth [H,W] and marker [2,N,2]')
    source_h, source_w = depth.shape
    depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_AREA)
    fixed, moved = markers
    displacement = moved - fixed
    vertices, weights = _interpolation_map(
        fixed.tobytes(), len(fixed), source_h, source_w, image_size)
    fields = np.einsum('nvc,nv->nc', displacement[vertices], weights).reshape(image_size, image_size, 2)
    return np.stack((np.clip((34. - depth) / 10., 0., 1.),
                     np.clip(fields[:, :, 0] / 40. + .5, 0., 1.),
                     np.clip(fields[:, :, 1] / 40. + .5, 0., 1.))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hdf5', type=Path)
    parser.add_argument('output', type=Path, help='Output .npy or .png')
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()
    with h5py.File(args.hdf5, 'r') as h5:
        prefix = f'tactile/{args.side}_tactile'
        result = convert(h5[f'{prefix}/depth'][args.step],
                         h5[f'{prefix}/marker'][args.step], args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == '.npy':
        np.save(args.output, result)
    elif args.output.suffix == '.png':
        cv2.imwrite(str(args.output), cv2.cvtColor(
            (result.transpose(1, 2, 0) * 255).round().astype(np.uint8),
            cv2.COLOR_RGB2BGR))
    else:
        parser.error('output must end with .npy or .png')
    print(f'Saved {args.output}: shape={result.shape}, range=({result.min():.3f}, {result.max():.3f})')


if __name__ == '__main__':
    main()
