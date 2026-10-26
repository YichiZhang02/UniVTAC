"""30 mm camera-depth baseline, hybrid displacement encoding, and cached TPS."""
from functools import lru_cache

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


NONLINEAR_DEPTH_DEFORM_SPEC = {
    'version': 'nonlinear_30mm_5mm_tps_v1',
    'reference_depth_mm': 30.0,
    'depth_range_mm': 5.0,
    'depth_source': 'camera_depth',
    'deform_linear_range_px': 20.0,
    'deform_linear_output_range': [0.2, 0.8],
    'deform_tail': 'rational',
    'deform_tail_scale_px': 40.0 / 3.0,
    'zero_displacement': 0.5,
    'interpolation': 'thin_plate_spline',
    'interpolation_coordinates': 'reference',
    'tps_smoothing': 0.0,
    'tps_degree': 1,
    'tps_merge_tolerance_px': 0.001,
    'tps_duplicate_reduction': 'mean',
}


def encode_displacement(displacement_px):
    """Linear on [-20,20]; rational tails with matching value and slope.

    f(x) = 0.5 + 0.015*x for |x| <= 20.
    Outside, f(x) = 0.5 + sign(x)*(0.5 - 0.2/(1+(|x|-20)/(40/3))).
    """
    field = np.asarray(displacement_px, dtype=np.float32)
    if not np.isfinite(field).all():
        raise ValueError('Displacements must be finite')
    magnitude = np.abs(field)
    tail = 0.5 - 0.2 / (1. + np.maximum(magnitude - 20., 0.) / (40. / 3.))
    signed_magnitude = np.where(magnitude <= 20., 0.015 * magnitude, tail)
    return (0.5 + np.sign(field) * signed_magnitude).astype(np.float32)


def merge_tps_controls(fixed):
    """Use the same duplicate reduction for CPU and batched CUDA evaluation."""
    count = len(fixed)
    # Old 64-marker training arrays pad a 63-marker grid by repeating a point,
    # sometimes with float32 roundoff. TPS requires distinct control points.
    parent = np.arange(count)

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in sorted(cKDTree(fixed).query_pairs(.001)):
        parent[root(b)] = root(a)
    _, group = np.unique([root(i) for i in range(count)], return_inverse=True)
    sizes = np.bincount(group)
    reduction = np.zeros((len(sizes), count), dtype=np.float64)
    reduction[group, np.arange(count)] = 1. / sizes[group]
    controls = reduction @ fixed.astype(np.float64)
    return controls, reduction


@lru_cache(maxsize=8)
def _tps_interpolation_map(fixed_bytes, count, source_h, source_w, image_size):
    """Cache dense TPS weights, merging duplicate controls within .001 pixels."""
    fixed = np.frombuffer(fixed_bytes, dtype=np.float32).reshape(count, 2)
    controls, reduction = merge_tps_controls(fixed)
    scale = float(max(source_h, source_w))
    xs = (np.arange(image_size, dtype=np.float64) + .5) * source_w / image_size
    ys = (np.arange(image_size, dtype=np.float64) + .5) * source_h / image_size
    xx, yy = np.meshgrid(xs, ys)
    query = np.column_stack((xx.ravel(), yy.ravel())) / scale
    interpolator = RBFInterpolator(controls / scale,
                                  reduction, kernel='thin_plate_spline',
                                  smoothing=0., degree=1)
    weights = interpolator(query).astype(np.float32)
    weights.setflags(write=False)
    return weights


def _convert_tps(depth, markers, image_size, displacement_encoder):
    """Return float32 [3,H,W]; depth is always raw camera depth in mm.

    Saved press_depth is deliberately not used: it can carry a different
    baseline. Reference/current markers must have matching stable ordering.
    """
    depth = np.asarray(depth, dtype=np.float32)
    markers = np.asarray(markers, dtype=np.float32)
    if depth.ndim != 2 or not depth.size:
        raise ValueError('Expected nonempty depth [H,W]')
    if markers.ndim != 3 or markers.shape[0] != 2 or markers.shape[-1] != 2 or markers.shape[1] < 3:
        raise ValueError('Expected markers [2,N,2] with at least 3 non-collinear markers')
    if not np.isfinite(depth).all() or not np.isfinite(markers).all():
        raise ValueError('Depth and markers must be finite')
    if not isinstance(image_size, (int, np.integer)) or image_size <= 0:
        raise ValueError('image_size must be a positive integer')
    resized = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_AREA)
    r = np.clip((30. - resized) / 5., 0., 1.)
    fixed, current = markers
    weights = _tps_interpolation_map(fixed.tobytes(), len(fixed), *depth.shape, image_size)
    field = (weights @ (current - fixed)).reshape(image_size, image_size, 2)
    gb = displacement_encoder(field).transpose(2, 0, 1)
    return np.concatenate((r[None], gb), axis=0).astype(np.float32)


def convert_nonlinear(depth, markers, image_size=224):
    """Original nonlinear encoding with shared camera-depth/TPS preprocessing."""
    return _convert_tps(depth, markers, image_size, encode_displacement)
