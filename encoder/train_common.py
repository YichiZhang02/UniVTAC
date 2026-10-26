"""Shared training loop for convolutional ResNet and ViT tactile encoders."""
import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from vit_methods import METHODS
from modality import INPUT_CHANNELS, read_tactile
from linear_depth_deform import DEPTH_DEFORM_SPECS, depth_deform_spec
from modality import tactile_from_arrays
from image_io import require_rgb_model
from pretrained_registry import history_indices, result_root
from checkpoint_io import save_checkpoint
from encoder.tactile_preprocessing import (
    collate_tactile_tps, initialize_data_worker, prepare_encoder_batch,
    prepare_tactile_tps, resolve_tactile_preprocessing,
)


class TactileFrames(Dataset):
    def __init__(self, files, limit=None, temporal=False, seed=42,
                 input_mode='marker_rgb', lengths=None, action_conditioned=False,
                 future_horizon=1, action_mean=None, action_std=None, num_frames=1,
                 tactile_preprocessing='cpu'):
        self.temporal = temporal
        self.action_conditioned = action_conditioned
        self.future_horizon = future_horizon
        self.action_mean = action_mean
        self.action_std = action_std
        self.input_mode = input_mode
        self.num_frames = num_frames
        self.tactile_preprocessing = tactile_preprocessing
        self.rows = []
        self.handles = {}
        self.mapped_arrays = {}
        for path in files:
            cached = lengths.get(str(path)) if lengths else None
            with h5py.File(path, 'r') if cached is None else nullcontext() as h5:
                for side in ('left', 'right'):
                    kind = ('depth' if input_mode in DEPTH_DEFORM_SPECS else 'marker' if input_mode == 'marker_only'
                            else 'rgb' if input_mode == 'rgb_only' else 'rgb_marker')
                    key = f'tactile/{side}_tactile/{kind}'
                    if cached is not None:
                        length = cached.get(side, {}).get(kind)
                        if length is None:
                            continue
                    else:
                        if key not in h5:
                            continue
                        length = len(h5[key])
                    usable = length - (future_horizon if action_conditioned else int(temporal))
                    self.rows.extend((str(path), side, t) for t in range(max(0, usable)))
        if limit and len(self.rows) > limit:
            self.rows = random.Random(seed).sample(self.rows, limit)
        if not self.rows:
            raise ValueError('No tactile frames found')

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['handles'] = {}
        state['mapped_arrays'] = {}
        return state

    def _read(self, path, side, step):
        if path not in self.handles:
            self.handles[path] = h5py.File(path, 'r')
        h5 = self.handles[path]
        if self.input_mode in DEPTH_DEFORM_SPECS:
            key = (path, side)
            if key not in self.mapped_arrays:
                prefix = f'tactile/{side}_tactile'
                names = ['depth', 'marker_full' if f'{prefix}/marker_full' in h5 else 'marker']
                if f'{prefix}/press_depth' in h5:
                    names.append('press_depth')
                datasets = {name: h5[f'{prefix}/{name}'] for name in names}
                if all(ds.chunks is None and ds.id.get_offset() is not None
                       for ds in datasets.values()):
                    self.mapped_arrays[key] = {
                        name: np.memmap(path, dtype=ds.dtype, mode='r',
                                        offset=ds.id.get_offset(), shape=ds.shape)
                        for name, ds in datasets.items()}
                else:
                    self.mapped_arrays[key] = None
            arrays = self.mapped_arrays[key]
            if arrays is not None:
                return tactile_from_arrays({name: value[step] for name, value in arrays.items()},
                                           self.input_mode)
        return read_tactile(h5, side, step, self.input_mode)

    def _joint_action(self, path, step):
        if path not in self.handles:
            self.handles[path] = h5py.File(path, 'r')
        joint = np.asarray(self.handles[path]['embodiment/joint'][step, :8], dtype=np.float32)
        return (joint - self.action_mean) / self.action_std

    def __getitem__(self, index):
        path, side, step = self.rows[index]
        if self.tactile_preprocessing == 'cuda':
            if path not in self.handles:
                self.handles[path] = h5py.File(path, 'r')
            h5 = self.handles[path]
            indices = history_indices(step, self.num_frames, 2) if not self.action_conditioned else [step]
            packed = {'images': [prepare_tactile_tps(h5, side, t, 224) for t in indices]}
            if self.action_conditioned or self.temporal:
                future = step + (self.future_horizon if self.action_conditioned else 1)
                packed['future_images'] = [prepare_tactile_tps(h5, side, future, 224)]
            if self.action_conditioned:
                packed['action'] = torch.from_numpy(self._joint_action(path, step + self.future_horizon))
            return packed
        if self.num_frames > 1 and not self.action_conditioned:
            image = torch.stack([self._read(path, side, t) for t in history_indices(step, self.num_frames, 2)])
        else:
            image = self._read(path, side, step)
        if self.action_conditioned:
            future = self._read(path, side, step + self.future_horizon)
            action = self._joint_action(path, step + self.future_horizon)
            return image, future, torch.from_numpy(action)
        return (image, self._read(path, side, step + 1)) if self.temporal else image


def joint_action_stats(files):
    """Project convention: action at t is the commanded absolute joint target at t+1."""
    chunks = []
    for path in files:
        with h5py.File(path, 'r') as h5:
            chunks.append(np.asarray(h5['embodiment/joint'][:, :8], dtype=np.float32))
    values = np.concatenate(chunks, axis=0)
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-2)


def probe_indices(data, per_class, seed, class_to_id):
    groups = {}
    for index, (path, _, _) in enumerate(data.rows):
        label = Path(path).parent.parent.name
        groups.setdefault(label, []).append(index)
    rng = random.Random(seed)
    picked = []
    labels = []
    for label in sorted(groups):
        selected = rng.sample(groups[label], min(per_class, len(groups[label])))
        picked.extend(selected)
        labels.extend([class_to_id[label]] * len(selected))
    return picked, torch.tensor(labels, dtype=torch.long)


class FileChunkBatchSampler(Sampler):
    """Mix small chunks from several HDF5 files in each depth batch."""
    def __init__(self, data, batch_size, seed, chunk_size=8):
        self.groups = {}
        for index, (path, _, _) in enumerate(data.rows):
            self.groups.setdefault(path, []).append(index)
        self.batch_size = batch_size
        self.chunk_size = min(chunk_size, batch_size)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return sum(map(len, self.groups.values())) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        chunks = []
        for group in self.groups.values():
            indices = group.copy()
            rng.shuffle(indices)
            chunks.extend(indices[start:start + self.chunk_size]
                          for start in range(0, len(indices), self.chunk_size))
        rng.shuffle(chunks)
        batch = []
        for chunk in chunks:
            batch.extend(chunk)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]


@torch.no_grad()
def knn_accuracy(encoder, train_probe, train_labels, val_probe, val_labels,
                 device, precision, input_mode=None):
    def embeddings(loader):
        chunks = []
        for batch in loader:
            batch = prepare_encoder_batch(batch, device, input_mode)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            with torch.autocast(device_type=device.type,
                                enabled=device.type == 'cuda' and precision == 'bf16',
                                dtype=torch.bfloat16):
                features = encoder(images)
            chunks.append(nn.functional.normalize(features.float(), dim=-1))
        return torch.cat(chunks)

    train_features = embeddings(train_probe)
    val_features = embeddings(val_probe)
    labels = train_labels.to(device)
    similarities = val_features @ train_features.T
    nearest = labels[similarities.topk(min(5, len(labels)), dim=1).indices]
    votes = nn.functional.one_hot(nearest, num_classes=int(max(train_labels.max(), val_labels.max())) + 1).sum(1)
    predicted = votes.argmax(dim=1)
    return float((predicted == val_labels.to(device)).float().mean())


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    args.tactile_preprocessing = resolve_tactile_preprocessing(
        args.tactile_preprocessing, args.input_mode, args.device)
    use_cuda_tps = args.tactile_preprocessing == 'cuda'
    files = sorted(args.data_root.glob('*/hdf5/*.hdf5'))
    if not files:
        raise FileNotFoundError(f'No contact HDF5 files under {args.data_root}')
    if not 0 < args.val_fraction < 1:
        raise ValueError('val_fraction must be between 0 and 1')
    if len(files) < 2:
        raise ValueError('At least two HDF5 files are required for a leak-free train/val split')
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / 'encoder_results/.contact_lengths.json'
    lengths = None
    if args.data_root.resolve() == (project_root / 'resources/data/contact').resolve() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('files') == [str(path) for path in files]:
            lengths = manifest['lengths']
    shuffled_files = files.copy()
    random.Random(args.seed).shuffle(shuffled_files)
    val_file_count = max(1, min(len(files) - 1, round(len(files) * args.val_fraction)))
    val_files = shuffled_files[:val_file_count]
    train_files = shuffled_files[val_file_count:]
    train_limit = round(args.samples * (1 - args.val_fraction)) if args.samples else None
    val_limit = args.samples - train_limit if args.samples else None
    action_conditioned = args.method == 'V-JEPA'
    action_mean, action_std = joint_action_stats(train_files) if action_conditioned else (None, None)
    train_data = TactileFrames(train_files, train_limit, action_conditioned, args.seed,
                               args.input_mode, lengths, action_conditioned,
                               args.future_horizon, action_mean, action_std,
                               tactile_preprocessing=args.tactile_preprocessing)
    val_data = TactileFrames(val_files, val_limit, action_conditioned, args.seed + 1,
                             args.input_mode, lengths, action_conditioned,
                             args.future_horizon, action_mean, action_std,
                             tactile_preprocessing=args.tactile_preprocessing)
    preprocessing_options = (dict(collate_fn=collate_tactile_tps, worker_init_fn=initialize_data_worker)
                             if use_cuda_tps else {})
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True, persistent_workers=args.workers > 0, **preprocessing_options)
    if args.input_mode in DEPTH_DEFORM_SPECS:
        batch_sampler = FileChunkBatchSampler(train_data, args.batch_size, args.seed)
        loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0, **preprocessing_options)
        val_data.rows.sort()
    else:
        loader = DataLoader(train_data, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(args.seed), **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    if not len(loader):
        raise ValueError('Encoder training requires at least 64 training anchors')
    train_probe = val_probe = None
    if args.checkpoint_metric == 'knn':
        if args.probe_per_class_train < 1 or args.probe_per_class_val < 1:
            raise ValueError('kNN checkpoint selection requires positive probe samples per class')
        classes = sorted({Path(path).parent.parent.name for path in files})
        class_to_id = {name: index for index, name in enumerate(classes)}
        train_pick, train_labels = probe_indices(
            train_data, args.probe_per_class_train, args.seed + 100, class_to_id)
        val_pick, val_labels = probe_indices(
            val_data, args.probe_per_class_val, args.seed + 101, class_to_id)
        probe_options = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True, **preprocessing_options)
        train_probe = DataLoader(Subset(train_data, train_pick), shuffle=False, **probe_options)
        val_probe = DataLoader(Subset(val_data, val_pick), shuffle=False, **probe_options)
    device = torch.device(args.device)
    model = METHODS[args.method](args.size, INPUT_CHANNELS[args.input_mode],
                                 **args.model_options).to(device)
    if args.multi_gpu and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer_class = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[args.optimizer]
    optimizer = optimizer_class((p for p in model.parameters() if p.requires_grad),
                                lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    best_probe_accuracy = -1.0
    stale_epochs = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if args.input_mode in ('rgb_only', 'marker_rgb'):
            require_rgb_model(saved)
        for key, expected in (('method', args.method), ('size', args.size),
                              ('input_mode', args.input_mode)):
            if saved.get(key) != expected:
                raise ValueError(f'Resume checkpoint {key}={saved.get(key)!r}, expected {expected!r}')
        if args.input_mode in DEPTH_DEFORM_SPECS and saved.get('depth_deform_encoding') != depth_deform_spec(args.input_mode):
            raise ValueError('Resume requires matching depth_deform normalization; start a new run for the new encoding')
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(saved['training_model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch = saved['epoch']
        global_step = saved['step']
        best_val_loss = saved.get('best_val_loss', float('inf'))
        best_probe_accuracy = saved.get('best_probe_accuracy', -1.0)
        stale_epochs = saved.get('stale_epochs', 0)
    family = 'resnet' if args.method == 'ResNet' else 'vit'
    out = args.output_dir or (args.resume.parent if args.resume else None) or (result_root(args.input_mode, 'univtac', args.method, args.size) /
                              datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(pretrained_mode='univtac', num_frames=1, frame_stride=1)
    config["color_order"] = "RGB"
    if args.input_mode in DEPTH_DEFORM_SPECS:
        config['depth_deform_encoding'] = depth_deform_spec(args.input_mode)
    config.update(data_root=str(args.data_root), output_dir=str(out),
                  count=len(train_data) + len(val_data),
                  train_count=len(train_data), val_count=len(val_data),
                  train_file_count=len(train_files), val_file_count=len(val_files),
                  train_files=[str(path) for path in train_files],
                  val_files=[str(path) for path in val_files], split_strategy='file',
                  data_sampling='file_chunks_8' if args.input_mode in DEPTH_DEFORM_SPECS
                                else 'random_frames')
    if action_conditioned:
        config['action_mean'] = action_mean.tolist()
        config['action_std'] = action_std.tolist()
    if args.config:
        config['config'] = str(args.config)
    if args.resume:
        config['resume'] = str(args.resume)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    log = out / 'metrics.jsonl'
    print(f'method={args.method} size={family}-{args.size} input={args.input_mode} '
          f'train={len(train_data)} val={len(val_data)} split=file '
          f'optimizer={args.optimizer} precision={args.precision} output={out}', flush=True)
    steps_per_epoch = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        previous_end = time.perf_counter()
        for step, batch in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            if device.type == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            data_wait = started - previous_end
            if global_step < warmup_steps:
                factor = (global_step + 1) / max(1, warmup_steps)
            else:
                fraction = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                factor = args.min_lr_ratio + (1 - args.min_lr_ratio) * .5 * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * factor
            batch = prepare_encoder_batch(batch, device, args.input_mode)
            if not isinstance(batch, (list, tuple)):
                batch = [batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                enabled=device.type == 'cuda' and args.precision == 'bf16',
                                dtype=torch.bfloat16):
                loss, metrics = model(*batch)
                loss = loss.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {args.method} loss at step {global_step}: {loss}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            core = model.module if isinstance(model, nn.DataParallel) else model
            if hasattr(core, 'update_teacher'):
                progress = min(1., (global_step + 1) / total_steps)
                momentum = 1 - (1 - args.teacher_momentum) * .5 * (1 + math.cos(math.pi * progress))
                core.update_teacher(momentum)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            previous_end = time.perf_counter()
            global_step += 1
            record = {'epoch': epoch + 1, 'step': global_step, 'loss': float(loss.detach()),
                      'lr': optimizer.param_groups[0]['lr'],
                      'batch_size': len(batch[0]),
                      'step_seconds': round(elapsed, 3),
                      'data_wait_seconds': round(data_wait, 3),
                      'batch_seconds': round(elapsed + data_wait, 3),
                      'samples_per_second': round(len(batch[0]) / elapsed, 1),
                      **{key: float(value.float().mean()) for key, value in metrics.items()}}
            if hasattr(core, 'update_teacher'):
                record['teacher_momentum'] = momentum
            if device.type == 'cuda':
                record['gpu_peak_gb'] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                                         for i in range(torch.cuda.device_count() if args.multi_gpu else 1)]
            with log.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        val_metric_sums = {}
        rng_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            torch.manual_seed(args.seed + 100000)
            for val_step, batch in enumerate(val_loader):
                if args.max_val_steps and val_step >= args.max_val_steps:
                    break
                batch = prepare_encoder_batch(batch, device, args.input_mode)
                if not isinstance(batch, (list, tuple)):
                    batch = [batch]
                with torch.autocast(device_type=device.type,
                                    enabled=device.type == 'cuda' and args.precision == 'bf16',
                                    dtype=torch.bfloat16):
                    val_loss, val_metrics = model(*batch)
                    val_loss = val_loss.mean()
                if not torch.isfinite(val_loss):
                    raise FloatingPointError(f'Nonfinite validation loss for {args.method}')
                val_loss_sum += float(val_loss) * len(batch[0])
                for key, value in val_metrics.items():
                    val_metric_sums[key] = val_metric_sums.get(key, 0.0) + float(value.float().mean()) * len(batch[0])
                val_seen += len(batch[0])
        val_loss = val_loss_sum / val_seen
        best_val_loss = min(best_val_loss, val_loss)
        probe_accuracy = None
        if train_probe is not None:
            core = model.module if isinstance(model, nn.DataParallel) else model
            probe_accuracy = knn_accuracy(core.encoder, train_probe, train_labels,
                                          val_probe, val_labels, device, args.precision, args.input_mode)
            improved = probe_accuracy > best_probe_accuracy
            if improved:
                best_probe_accuracy = probe_accuracy
        else:
            improved = val_loss <= best_val_loss
        if improved:
            stale_epochs = 0
        else:
            stale_epochs += 1
        validation_record = {'epoch': epoch + 1, 'step': global_step,
                             'val_loss': val_loss, 'best_val_loss': best_val_loss,
                             'val_samples': val_seen, 'best': improved,
                             'knn_accuracy': probe_accuracy,
                             'best_knn_accuracy': best_probe_accuracy if train_probe is not None else None,
                             **{f'val_{key}': value / val_seen
                                for key, value in val_metric_sums.items()}}
        with log.open('a') as handle:
            handle.write(json.dumps(validation_record) + '\n')
        print(json.dumps(validation_record), flush=True)
        core = model.module if isinstance(model, nn.DataParallel) else model
        if not args.no_save:
            checkpoint = {'format_version': 3, 'color_order': 'RGB', 'method': args.method, 'size': args.size,
                          'pretrained_mode': 'univtac', 'num_frames': 1, 'frame_stride': 1,
                          'architecture': family,
                          'image_size': 224, 'input_mode': args.input_mode,
                          'in_chans': INPUT_CHANNELS[args.input_mode],
                          'depth_deform_encoding': depth_deform_spec(args.input_mode) if args.input_mode in DEPTH_DEFORM_SPECS else None,
                          'encoder': core.encoder.state_dict(),
                          'training_model': core.state_dict(), 'optimizer': optimizer.state_dict(),
                          'epoch': epoch + 1, 'step': global_step,
                          'val_loss': val_loss, 'best_val_loss': best_val_loss,
                          'knn_accuracy': probe_accuracy,
                          'best_probe_accuracy': best_probe_accuracy,
                          'stale_epochs': stale_epochs}
            save_checkpoint(checkpoint, out / 'last.pth')
            if improved:
                if args.method == 'VAE':
                    # Keep the matching encoder/decoder for best-epoch reconstruction.
                    save_checkpoint(checkpoint, out / 'best.pth')
                save_checkpoint({key: checkpoint[key] for key in ('format_version', 'pretrained_mode', 'num_frames', 'frame_stride', 'method', 'size', 'architecture',
                            'image_size', 'input_mode', 'in_chans', 'color_order', 'depth_deform_encoding', 'encoder', 'epoch', 'step',
                            'val_loss', 'best_val_loss', 'knn_accuracy',
                            'best_probe_accuracy')}, out / 'encoder.pth')
            print(f'Saved {out / "last.pth"}', flush=True)
        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            print(f'Early stopping after {stale_epochs} epochs without validation improvement', flush=True)
            break
    return out


def main(default_method=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=default_method is None, default=default_method, choices=METHODS)
    parser.add_argument('--size', default='S', choices=['S', 'B'])
    parser.add_argument('--input-mode', default='marker_rgb', choices=INPUT_CHANNELS)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent.parent / 'resources/data/contact')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, choices=[64])
    parser.add_argument('--workers', type=int)
    parser.add_argument('--tactile-preprocessing', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--warmup-epochs', type=int)
    parser.add_argument('--min-lr-ratio', type=float)
    parser.add_argument('--grad-clip', type=float)
    parser.add_argument('--teacher-momentum', type=float)
    parser.add_argument('--future-horizon', type=int)
    parser.add_argument('--optimizer', choices=['Adam', 'AdamW'])
    parser.add_argument('--precision', choices=['fp32', 'bf16'])
    parser.add_argument('--val-fraction', type=float)
    parser.add_argument('--max-val-steps', type=int)
    parser.add_argument('--early-stop-patience', type=int)
    parser.add_argument('--checkpoint-metric', choices=['val_loss', 'knn'])
    parser.add_argument('--probe-per-class-train', type=int)
    parser.add_argument('--probe-per-class-val', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--multi-gpu', action='store_true', default=None)
    parser.add_argument('--max-steps', type=int, default=None, help='Limit optimizer steps per epoch for smoke tests')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Resume from a last.pth checkpoint')
    parser.add_argument('--no-save', action='store_true', help='Skip checkpoints for throughput measurements')
    args = parser.parse_args()
    args.config = args.config or Path(__file__).resolve().parent / args.method / 'config.yml'
    settings = yaml.safe_load(args.config.read_text())[args.size]
    settings.update(settings.pop('input_overrides', {}).get(args.input_mode, {}))
    args.model_options = settings.pop('model', {})
    defaults = {'samples': 100000, 'epochs': 5, 'batch_size': 64, 'workers': 8,
                'lr': 1e-4, 'weight_decay': .05, 'warmup_epochs': 1,
                'min_lr_ratio': .1, 'grad_clip': 1., 'teacher_momentum': .996,
                'multi_gpu': False, 'optimizer': 'AdamW', 'precision': 'bf16',
                'val_fraction': .2, 'max_val_steps': None,
                'early_stop_patience': 0, 'checkpoint_metric': 'val_loss',
                'probe_per_class_train': 0, 'probe_per_class_val': 0,
                'future_horizon': 1}
    defaults.update(settings)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.batch_size != 64:
        parser.error('All encoder training batches must be 64, including config overrides')
    run(args)


if __name__ == '__main__':
    main()
