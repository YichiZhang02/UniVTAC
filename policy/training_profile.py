"""Opt-in synchronized training stage timings; disabled in ordinary training."""
import json
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


class TrainingStageProfiler:
    def __init__(self, model, output, warmup):
        self.output, self.warmup = Path(output), warmup
        self.rows, self.current = [], None
        self.wrap(model, '_encode_prefix', 'prefix_forward_ms')
        tactile = getattr(model, 'tactile_encoder', None)
        if tactile is not None:
            self.wrap(tactile, 'forward_flat', 'tactile_forward_ms')
            parameters = tuple(p for p in tactile.parameters() if p.requires_grad)
            if parameters:
                self._tactile_grad_handle = torch.autograd.graph.register_multi_grad_hook(
                    parameters, self.end_tactile_backward)
        qwen = getattr(model, 'qwen_vl', None)
        if qwen is not None:
            self.wrap(qwen, 'forward', 'qwen_forward_ms')
            self.wrap(qwen, 'forward_with_prefix_tokens', 'qwen_forward_ms')
            self.wrap(qwen, 'build_qwenvl_inputs', 'qwen_input_preparation_ms')
        action = getattr(model, 'action_head', None)
        if action is not None:
            self.wrap(action, 'forward', 'action_forward_ms')

    @staticmethod
    def event():
        value = torch.cuda.Event(enable_timing=True)
        value.record()
        return value

    def wrap(self, module, method, name):
        if not hasattr(module, method):
            return
        original = getattr(module, method)

        def timed(*args, **kwargs):
            if self.current is None:
                return original(*args, **kwargs)
            start = self.event()
            result = original(*args, **kwargs)
            self.current['scopes'].append((name, start, self.event()))
            if name == 'tactile_forward_ms' and result.requires_grad:
                result.register_hook(self.start_tactile_backward)
            return result

        setattr(module, method, timed)

    def start_tactile_backward(self, gradient):
        if self.current is not None:
            self.current['tactile_backward_start'] = self.event()
        return gradient

    def end_tactile_backward(self, gradients):
        # This is the branch's gradient-ready interval, including any work
        # interleaved with it. DDP can overlap it; it is a subdivision of the
        # backward interval, not an additional additive communication timing.
        if self.current is not None and 'tactile_backward_start' in self.current:
            self.current['scopes'].append(('tactile_backward_ms',
                self.current.pop('tactile_backward_start'), self.event()))

    def begin(self, step):
        self.current = dict(step=step, wall_start=time.perf_counter(), marks={}, scopes=[])

    def mark(self, name):
        self.current['marks'][name] = self.event()

    def finish(self, data_wait_ms):
        self.mark('end')
        row = self.current
        row['marks']['end'].synchronize()
        values = dict(step=row['step'], data_wait_ms=data_wait_ms,
                      step_wall_ms=(time.perf_counter() - row['wall_start']) * 1000)
        marks = row['marks']
        for name, start, end in (
            ('host_to_device_ms', 'h2d_start', 'h2d_end'),
            ('tps_ms', 'h2d_end', 'tps_end'),
            ('forward_ms', 'tps_end', 'forward_end'),
            ('backward_ddp_ms', 'forward_end', 'backward_end'),
            ('clip_ms', 'backward_end', 'clip_end'),
            ('optimizer_ms', 'clip_end', 'optimizer_end'),
        ):
            values[name] = marks[start].elapsed_time(marks[end])
        for name, start, end in row['scopes']:
            values[name] = values.get(name, 0.) + start.elapsed_time(end)
        primary = ('data_wait_ms', 'host_to_device_ms', 'tps_ms', 'forward_ms',
                   'backward_ddp_ms', 'clip_ms', 'optimizer_ms')
        values['other_ms'] = values['step_wall_ms'] - sum(values[k] for k in primary)
        self.rows.append(values)
        self.current = None

    def save(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        rows = self.rows[self.warmup:]
        summary = {}
        if rows:
            for key in rows[0]:
                if key == 'step':
                    continue
                values = sorted(row[key] for row in rows)
                summary[key] = dict(mean=statistics.mean(values), median=statistics.median(values),
                                    min=values[0], max=values[-1],
                                    p90=values[min(len(values)-1, int(.9 * len(values)))])
        report = dict(rank=rank, world_size=world, gpu=torch.cuda.get_device_name(),
                      torch=torch.__version__, warmup_steps=self.warmup, measured_steps=len(rows),
                      peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                      summary=summary, rows=self.rows)
        (self.output / f'stage_profile_rank{rank}.json').write_text(json.dumps(report, indent=2))
        reports = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(report, reports, dst=0)
        else:
            reports = [report]
        if rank == 0:
            (self.output / 'stage_profile.json').write_text(json.dumps(reports, indent=2))
