"""Physical simulation-step contracts shared by pretraining and policy inference."""
from collections import deque
import numpy as np
import torch


def validate_simulation_spacing(h5, frame_stride, simulation_step_stride):
    # Historical data recorded physics steps as `step` during 120 Hz collection.
    # New observations additionally expose the unambiguous physical-step counter.
    key = 'physics_step' if 'physics_step' in h5 else 'step'
    steps = np.asarray(h5[key]).reshape(-1)
    delta = np.diff(steps)
    if len(delta) and (delta <= 0).any():
        raise ValueError(f'Nonchronological contact trajectory: {h5.filename}')
    if len(delta) and not np.all(delta == delta[0]):
        raise ValueError(f'Discontinuous contact trajectory: {h5.filename}')
    if simulation_step_stride is not None and len(delta) and not np.all(delta * frame_stride == simulation_step_stride):
        raise ValueError(f'{h5.filename}: saved spacing {delta[0]} * frame_stride {frame_stride} '
                         f'!= required simulation_step_stride {simulation_step_stride}')
    return steps


class SimulationTactileHistory:
    """Keep distinct real observations separated by exactly N physical steps."""
    def __init__(self, num_frames=4, simulation_step_stride=2):
        self.num_frames = num_frames
        self.simulation_step_stride = simulation_step_stride
        self.frames = deque(maxlen=num_frames)
        self.steps = deque(maxlen=num_frames)

    @property
    def ready(self):
        return len(self.frames) == self.num_frames

    def append(self, frame, physics_step):
        step = int(physics_step)
        if self.steps and step == self.steps[-1]:
            return  # Re-encoding an observation must not advance time.
        if self.steps and step - self.steps[-1] != self.simulation_step_stride:
            raise ValueError(f'Inference tactile interval is {step-self.steps[-1]} simulation steps; '
                             f'encoder requires {self.simulation_step_stride}')
        self.steps.append(step)
        self.frames.append(frame)

    def clip(self, *, require_complete=True):
        if require_complete and not self.ready:
            raise ValueError('Four distinct real tactile observations are required; warm up the history first')
        return torch.stack(list(self.frames))
