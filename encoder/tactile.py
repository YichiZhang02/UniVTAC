"""Shared recorded/live depth_deform_radial preprocessing.

Marker RGB is retained for inspection. Metric depth and paired marker coordinates
are required; these quantities cannot be recovered exactly from RGB alone.
"""
from functools import lru_cache

import cv2
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree

REPRESENTATION = 'depth_deform_radial'
DEFAULT_INTERPOLATION = 'tps_bilinear'
DEFAULT_GRID = 28


def representation_spec(method=DEFAULT_INTERPOLATION, grid=DEFAULT_GRID):
    return dict(version='depth_deform_radial_v1', input_mode=REPRESENTATION,
                depth_source='camera_depth_mm', reference_depth_mm=30., depth_range_mm=5.,
                deform_range_px=30., deform_mapping='radial', zero_displacement=.5,
                interpolation=method, coarse_grid=grid, coordinates='reference_pixel_centers',
                duplicate_tolerance_px=.001, duplicate_reduction='mean')


def merge_controls(markers):
    markers = np.asarray(markers, dtype=np.float64)
    if markers.ndim != 3 or markers.shape[0] != 2 or markers.shape[2] != 2 or markers.shape[1] < 3:
        raise ValueError('Expected paired reference/current markers [2,N,2], N>=3')
    if not np.isfinite(markers).all():
        raise ValueError('Markers must be finite')
    fixed, current = markers
    parent = np.arange(len(fixed))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in sorted(cKDTree(fixed).query_pairs(.001)):
        parent[root(b)] = root(a)
    _, groups = np.unique([root(i) for i in range(len(fixed))], return_inverse=True)
    counts = np.bincount(groups)
    points = np.zeros((len(counts), 2))
    values = np.zeros_like(points)
    np.add.at(points, groups, fixed)
    np.add.at(values, groups, current-fixed)
    points /= counts[:, None]
    values /= counts[:, None]
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(0)) < 2:
        raise ValueError('At least three distinct, non-collinear marker controls required')
    return points, values


@lru_cache(maxsize=24)
def query_grid(source_hw, size):
    h, w = source_hw
    xx, yy = np.meshgrid((np.arange(size)+.5)*w/size,
                         (np.arange(size)+.5)*h/size)
    return np.stack((xx, yy), -1)


def dense_displacement(markers, source_hw, size=224, *, method=DEFAULT_INTERPOLATION,
                       grid=DEFAULT_GRID):
    points, values = merge_controls(markers)
    query = query_grid(tuple(source_hw), size)
    scale = max(source_hw)
    if method in ('tps', 'tps_bilinear'):
        fit = RBFInterpolator(points/scale, values, kernel='thin_plate_spline',
                              smoothing=0., degree=1)
        if method == 'tps':
            return fit(query.reshape(-1, 2)/scale).reshape(size, size, 2).astype(np.float32)
        if not isinstance(grid, int) or grid < 2:
            raise ValueError('Coarse TPS grid must be an integer >= 2')
        # Include final pixel centers at both ends, avoiding resize's half-pixel
        # edge mismatch. Exact separable bilinear avoids remap's 1/32 weight
        # quantization, preserving affine displacements across the whole image.
        xs = np.linspace(query[0, 0, 0], query[0, -1, 0], grid)
        ys = np.linspace(query[0, 0, 1], query[-1, 0, 1], grid)
        xx, yy = np.meshgrid(xs, ys)
        coarse = fit(np.stack((xx, yy), -1).reshape(-1, 2)/scale).reshape(grid, grid, 2).astype(np.float32)
        coords = np.linspace(0, grid-1, size, dtype=np.float32)
        lo = np.floor(coords).astype(np.int64)
        hi = np.minimum(lo+1, grid-1)
        fraction = (coords-lo).astype(np.float32)
        horizontal = coarse[:,lo]*(1-fraction)[None,:,None]+coarse[:,hi]*fraction[None,:,None]
        return horizontal[lo]*(1-fraction)[:,None,None]+horizontal[hi]*fraction[:,None,None]
    if method == 'linear':
        flat = query.reshape(-1, 2)
        result = LinearNDInterpolator(points, values)(flat)
        outside = ~np.isfinite(result).all(-1)
        if outside.any():
            # An affine trend plus IDW residual gives finite extrapolation and
            # exactly reproduces affine motion, unlike a zero/nearest boundary.
            basis = np.c_[np.ones(len(points)), points/scale]
            affine = np.linalg.lstsq(basis, values, rcond=None)[0]
            residual = values-basis@affine
            distance, indices = cKDTree(points).query(flat[outside], k=min(8, len(points)))
            weights = 1./np.maximum(distance, 1e-9)**2
            weights /= weights.sum(-1, keepdims=True)
            result[outside] = np.c_[np.ones(outside.sum()), flat[outside]/scale]@affine
            result[outside] += (weights[..., None]*residual[indices]).sum(1)
        return result.reshape(size, size, 2).astype(np.float32)
    raise ValueError(f'Unknown interpolation: {method}')


def convert(depth, markers, size=224, *, method=DEFAULT_INTERPOLATION, grid=DEFAULT_GRID):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all():
        raise ValueError('Expected finite camera depth [H,W] in millimetres')
    field = dense_displacement(markers, depth.shape, size, method=method, grid=grid)
    gain = .5/np.maximum(np.linalg.norm(field, axis=-1, keepdims=True), 30.)
    gb = .5+field*gain
    r = np.clip((30.-cv2.resize(depth, (size, size), interpolation=cv2.INTER_AREA))/5., 0., 1.)
    return torch.from_numpy(np.concatenate((r[None], gb.transpose(2, 0, 1)), 0).astype(np.float32))


def sensor_group(h5, side):
    for suffix in ('tactile', 'gsmini'):
        key = f'tactile/{side}_{suffix}'
        if key in h5:
            return h5[key]
    raise KeyError(f'Missing {side} tactile sensor')


def from_group(group, step, size=224, **kwargs):
    key = 'marker_full' if 'marker_full' in group else 'marker'
    return convert(group['depth'][step], group[key][step], size, **kwargs)


def from_observation(data, size=224, **kwargs):
    def array(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    markers = data['marker_full'] if 'marker_full' in data else data['marker']
    return convert(array(data['depth']), array(markers), size, **kwargs)
