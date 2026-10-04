"""Convert contact HDF5 observations to the four tactile encoder input modes."""
import cv2
import numpy as np
import torch
from depth_marker_to_deform import convert as depth_marker_to_deform


INPUT_CHANNELS = {'marker_only': 3, 'rgb_only': 3, 'marker_rgb': 3,
                  'depth_deform': 3}


def tactile_from_arrays(data, mode, image_size=224):
    """Encode raw RGB/depth/marker arrays identically for training and deployment."""
    if mode not in INPUT_CHANNELS:
        raise ValueError(f'Unknown tactile input mode: {mode}')
    if mode in ('rgb_only', 'marker_rgb'):
        key = 'rgb' if mode == 'rgb_only' else 'rgb_marker'
        image = np.asarray(data[key])
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    if mode == 'marker_only':
        markers = np.asarray(data['marker'])[1, :63]
        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
        for x, y in markers:
            cx = int(round(float(x) * image_size / 320))
            cy = int(round(float(y) * image_size / 240))
            if 0 <= cx < image_size and 0 <= cy < image_size:
                cv2.circle(canvas, (cx, cy), 2, 255, -1)
        # Replicate the marker raster into RGB; each channel contains the same dots.
        return torch.from_numpy(canvas.copy()).unsqueeze(0).repeat(3, 1, 1).float() / 255
    channels = depth_marker_to_deform(data['depth'], np.asarray(data['marker'])[:, :63], image_size)
    return torch.from_numpy(channels)


def read_tactile(h5, side, step, mode, image_size=224):
    prefix = f'tactile/{side}_tactile'
    if mode in ('rgb_only', 'marker_rgb'):
        key = 'rgb' if mode == 'rgb_only' else 'rgb_marker'
        raw = np.frombuffer(h5[f'{prefix}/{key}'][step], dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode {h5.filename}:{prefix}/{key}[{step}]')
        data = {key: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)}
    elif mode == 'marker_only':
        data = {'marker': h5[f'{prefix}/marker'][step]}
    elif mode == 'depth_deform':
        data = {'depth': h5[f'{prefix}/depth'][step], 'marker': h5[f'{prefix}/marker'][step]}
    else:
        raise ValueError(f'Unknown tactile input mode: {mode}')
    return tactile_from_arrays(data, mode, image_size)
