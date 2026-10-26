"""Small CPU TPS solves followed by dense, batched CUDA evaluation.

The CPU reference materializes a [H*W, markers] interpolation map for every
distinct reference grid. Recorded reference grids vary between frames, so
that cache rarely helps shuffled training. Transferring the two displacement
coefficients per control avoids rebuilding and transferring that dense map.
"""
from functools import lru_cache

import cv2
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from encoder.nonlinear_depth_deform import merge_tps_controls


def prepare_tactile_tps(h5, side, step, image_size):
    group = h5[f'tactile/{side}_tactile']
    depth = np.asarray(group['depth'][step], dtype=np.float32)
    markers = np.asarray(group['marker_full' if 'marker_full' in group else 'marker'][step], dtype=np.float32)
    if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all():
        raise ValueError('Expected finite, nonempty camera depth [H,W]')
    if markers.ndim != 3 or markers.shape[0] != 2 or markers.shape[-1] != 2 or markers.shape[1] < 3 or not np.isfinite(markers).all():
        raise ValueError('Expected finite markers [2,N,2], N >= 3')
    fixed, current = markers
    controls, reduction = merge_tps_controls(fixed)
    metric_scale = float(max(depth.shape))
    controls = controls / metric_scale
    rbf = RBFInterpolator(controls, reduction @ (current - fixed),
                          kernel='thin_plate_spline', smoothing=0., degree=1)
    # SciPy's fitted coefficients include its normalized affine polynomial.
    # Keeping the solve in SciPy also retains its invalid-grid checks.
    coefficients = rbf._coeffs
    resized = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return {
        'controls': torch.from_numpy(controls),
        'radial': torch.from_numpy(coefficients[:-3].copy()),
        'affine': torch.from_numpy(coefficients[-3:].copy()),
        'shift': torch.from_numpy(rbf._shift.copy()),
        'scale': torch.from_numpy(rbf._scale.copy()),
        'depth': torch.from_numpy(np.clip((30. - resized) / 5., 0., 1.)),
        'source_hw': torch.tensor(depth.shape, dtype=torch.int64),
    }


def collate_tactile_tps(samples):
    """Pad variable marker counts; keep ordinary observations on default_collate."""
    ordinary = [{k: v for k, v in sample.items() if not isinstance(v, list)} for sample in samples]
    batch = default_collate(ordinary)
    for key, value in samples[0].items():
        if not isinstance(value, list):
            continue
        frames = len(value)
        flat = [frame for sample in samples for frame in sample[key]]
        for field in value[0]:
            values = [frame[field] for frame in flat]
            tensor = pad_sequence(values, batch_first=True) if field in ('controls', 'radial') else torch.stack(values)
            batch[f'{key}.tps_{field}'] = tensor.reshape(len(samples), frames, *tensor.shape[1:])
    return batch


@lru_cache(maxsize=16)
def _query_grid(source_h, source_w, image_size, device):
    # Match the CPU reference's pixel-center coordinates and uniform metric.
    ys = (torch.arange(image_size, dtype=torch.float64, device=device) + .5) * source_h / image_size
    xs = (torch.arange(image_size, dtype=torch.float64, device=device) + .5) * source_w / image_size
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack((xx.flatten(), yy.flatten()), -1) / max(source_h, source_w)


@torch.no_grad()
def materialize_tactile_tps(batch, mode, *, frame_chunk=8):
    """Replace compact TPS fields with the reference [B,F,3,H,W] observation."""
    for name in list(batch):
        if not name.endswith('.tps_controls'):
            continue
        key = name[:-len('.tps_controls')]
        fields = {field: batch.pop(f'{key}.tps_{field}') for field in
                  ('controls', 'radial', 'affine', 'shift', 'scale', 'depth', 'source_hw')}
        b, frames, height, width = fields['depth'].shape
        if height != width:
            raise ValueError('TPS output must be square')
        # source_hw stays on CPU: inspecting small shape metadata must not
        # synchronize the GPU stream. Real sensor resolutions may differ.
        source_hw = fields.pop('source_hw').reshape(-1, 2).tolist()
        flat = {field: tensor.flatten(0, 1) for field, tensor in fields.items()}
        result = torch.empty((b * frames, 3, height, width), dtype=torch.float32, device=flat['depth'].device)
        result[:, 0] = flat['depth']
        for start in range(0, b * frames, frame_chunk):
            stop = min(start + frame_chunk, b * frames)
            grids = [_query_grid(h, w, height, result.device) for h, w in source_hw[start:stop]]
            query = torch.stack(grids)
            controls = flat['controls'][start:stop]
            distance2 = (query.square().sum(-1, keepdim=True)
                         + controls.square().sum(-1).unsqueeze(1)
                         - 2 * torch.bmm(query, controls.transpose(1, 2))).clamp_min_(0)
            kernel = .5 * distance2 * distance2.clamp_min(1e-300).log()
            field = torch.bmm(kernel, flat['radial'][start:stop])
            polynomial = (query - flat['shift'][start:stop, None]) / flat['scale'][start:stop, None]
            affine = flat['affine'][start:stop]
            field += affine[:, :1] + torch.bmm(polynomial, affine[:, 1:])
            if mode == 'depth_deform_nonlinear_radial':
                radius = torch.linalg.vector_norm(field, dim=-1)
                amplitude = torch.where(radius <= 20., .015 * radius,
                    .5 - .2 / (1. + (radius - 20.).clamp_min(0) / (40. / 3.)))
                gain = torch.where(radius > 0, amplitude / radius.clamp_min(1e-300), .015)
                encoded = (.5 + field * gain[..., None]).float()
            elif mode == 'depth_deform_nonlinear':
                # The original componentwise encoder rounds the field to fp32 first.
                field = field.float()
                magnitude = field.abs()
                mapped = torch.where(magnitude <= 20., .015 * magnitude,
                    .5 - .2 / (1. + (magnitude - 20.).clamp_min(0) / (40. / 3.)))
                encoded = .5 + field.sign() * mapped
            else:
                raise ValueError(f'Unsupported TPS input mode: {mode}')
            result[start:stop, 1:] = encoded.transpose(1, 2).reshape(stop-start, 2, height, width)
        batch[key] = result.reshape(b, frames, 3, height, width) if frames > 1 else result
    return batch


def initialize_data_worker(_worker_id):
    """Prevent OpenCV and BLAS pools multiplying across ranks and data workers."""
    cv2.setNumThreads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def resolve_tactile_preprocessing(requested, input_mode, device):
    """Select dense CUDA TPS only for supported representations/devices."""
    supported = (str(device).startswith('cuda') and input_mode in
                 ('depth_deform_nonlinear', 'depth_deform_nonlinear_radial'))
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError(f'Unknown tactile preprocessing mode: {requested}')
    if requested == 'cuda' and not supported:
        raise ValueError('CUDA TPS requires a CUDA device and a nonlinear TPS input mode')
    return 'cuda' if supported and requested != 'cpu' else 'cpu'


def prepare_encoder_batch(batch, device, input_mode=None):
    """Materialize compact encoder inputs on the main process's CUDA device.

    The output contract matches the CPU dataset: one image/clip tensor or a
    list containing current image, future image and optionally normalized action.
    """
    if isinstance(batch, dict):
        batch = {key: value.to(device, non_blocking=True)
                 if isinstance(value, torch.Tensor) and not key.endswith('.tps_source_hw') else value
                 for key, value in batch.items()}
        batch = materialize_tactile_tps(batch, input_mode)
        if 'future_images' in batch:
            values = [batch['images'], batch['future_images']]
            if 'action' in batch:
                values.append(batch['action'])
            return values
        return batch['images']
    if isinstance(batch, (tuple, list)):
        return [value.to(device, non_blocking=True) for value in batch]
    return batch.to(device, non_blocking=True)
