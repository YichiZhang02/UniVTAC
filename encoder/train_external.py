"""Pretrain external encoders from scratch or official weights on contact data."""
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'encoder'))
from encoder.external import ExternalEncoder, resize_frames
from encoder.pretrained_registry import result_root, validate_identity
from encoder.linear_depth_deform import DEPTH_DEFORM_SPECS, depth_deform_spec
from encoder.modality import INPUT_CHANNELS
from encoder.checkpoint_io import save_checkpoint
from train_common import TactileFrames
from encoder.tactile_preprocessing import (
    collate_tactile_tps, initialize_data_worker, prepare_encoder_batch,
    resolve_tactile_preprocessing,
)


class ExternalFrames(TactileFrames):
    """Match policy RGB background subtraction before external preprocessing."""
    def __init__(self, *args, background_offset=None, **kwargs):
        self.background_offset = background_offset
        self.baselines = {}
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        images = super().__getitem__(index)
        if self.background_offset is not None and self.input_mode in ('rgb_only', 'marker_rgb'):
            path, side, _ = self.rows[index]
            key = path, side
            if key not in self.baselines:
                self.baselines[key] = self._read(path, side, 0)
            images = (images-self.baselines[key]+self.background_offset).clamp(0, 1)
        return images


@torch.no_grad()
def calibrate_wan(encoder, loader, device, max_steps, input_mode=None):
    """Training-split posterior-mode statistics used by downstream token scaling."""
    encoder.eval()
    channels = encoder.model.config.z_dim
    sums = torch.zeros(2, channels, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for index, images in enumerate(loader):
        if index >= max_steps:
            break
        images = prepare_encoder_batch(images, device, input_mode)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
            latent = encoder.posterior(images).mode().double()
        sums[0] += latent.sum((0, 2, 3, 4))
        sums[1] += latent.square().sum((0, 2, 3, 4))
        count += latent.numel()/channels
    if dist.is_initialized():
        dist.all_reduce(sums)
        dist.all_reduce(count)
    mean = sums[0]/count
    std = (sums[1]/count-mean.square()).clamp_min(1e-8).sqrt()
    encoder.model.register_to_config(latents_mean=mean.tolist(), latents_std=std.tolist())
    encoder.metadata['model_config'] = dict(encoder.model.config)
    return {'anchors_per_rank_max': max_steps*loader.batch_size, 'latent_values_per_channel': int(count.item())}


class VisionAdaptation(nn.Module):
    """Masked teacher-feature distillation + tactile patch reconstruction.

    These are UniVTAC adaptation objectives, not reproductions of upstream training.
    Wan uses its pretrained decoder with reconstruction + KL instead.
    """
    def __init__(self, encoder, mask_ratio=.5, kl_weight=1e-6):
        super().__init__()
        self.encoder, self.mask_ratio, self.kl_weight = encoder, mask_ratio, kl_weight
        if encoder.method == 'Wan2.2-VAE':
            self.decoder = encoder._initial_decoder
            self.post_quant_conv = encoder._initial_post_quant_conv
            from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
            self.decoder_convs = sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
            from encoder.wan_recompute import enable_wan_recomputation
            enable_wan_recomputation(encoder.model.encoder)
            enable_wan_recomputation(self.decoder)
        else:
            self.teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
            dim = encoder.embed_dim
            self.predictor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.reconstruct = nn.Linear(dim, 3)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum=.996):
        if not hasattr(self, 'teacher'):
            return
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            if student.is_floating_point() or student.is_complex():
                teacher.lerp_(student, 1-momentum)
            else:
                # AnyTouch2 stores fixed integer position IDs as a Parameter.
                teacher.copy_(student)

    def forward(self, images):
        # DataParallel worker threads need an explicit bf16 context of their own.
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            loss, metrics = self._forward(images)
        return loss, dict(metrics, _sample_count=loss.new_tensor(len(images)))

    def _forward(self, images):
        if self.encoder.method == 'Wan2.2-VAE':
            # DataParallel shallow-copies Python attributes. Each replica owns its
            # cache counters before encode/decode mutate them on separate threads.
            self.encoder.model._cached_conv_counts = dict(self.encoder.model._cached_conv_counts)
            posterior = self.encoder.posterior(images)
            model = self.encoder.model
            # Decoder is registered on the training objective, never on the exported encoder.
            model.decoder, model.post_quant_conv = self.decoder, self.post_quant_conv
            model._cached_conv_counts['decoder'] = self.decoder_convs
            try:
                decoded = model.decode(posterior.sample()).sample
                if self.encoder.num_frames == 4:
                    decoded = decoded[:, :, 1:5]
            finally:
                model.decoder, model.post_quant_conv = None, None
                model._cached_conv_counts['decoder'] = 0
            target = resize_frames(images, self.encoder.hw).transpose(1, 2)*2-1
            reconstruction = F.mse_loss(decoded.float(), target.float())
            kl = posterior.kl().mean() / posterior.mean[0].numel()
            return reconstruction + self.kl_weight*kl, dict(reconstruction=reconstruction.detach(), kl=kl.detach())
        h, w = images.shape[-2:]
        mask = torch.rand(images.shape[0], 1, 14, 14, device=images.device) < self.mask_ratio
        mask = F.interpolate(mask.float(), (h, w), mode='nearest')
        if images.ndim == 5:
            mask = mask[:, None]
        masked = images*(1-mask) + .5*mask
        tokens = self.encoder.patch_tokens(masked)
        with torch.no_grad():
            target = self.teacher.patch_tokens(images)
        feature_loss = F.mse_loss(F.layer_norm(self.predictor(tokens).float(), (tokens.shape[-1],)),
                                 F.layer_norm(target.float(), (target.shape[-1],)))
        # Align target RGB patch means with the backbone's spatial and temporal lattice.
        patch = 16
        gh, gw = self.encoder.hw[0]//patch, self.encoder.hw[1]//patch
        gt = tokens.shape[1]//(gh*gw)
        source = images.transpose(1, 2) if images.ndim == 5 else images[:, :, None]
        pixels = F.adaptive_avg_pool3d(source, (gt, gh, gw)).flatten(2).transpose(1, 2)
        reconstruction = F.mse_loss(self.reconstruct(tokens).float(), pixels.float())
        loss = feature_loss + reconstruction
        return loss, dict(feature=feature_loss.detach(), reconstruction=reconstruction.detach())


def reduce_replica_loss(loss, metrics):
    """Sample-weighted reduction also handles unequal validation replica sizes."""
    counts = metrics.pop('_sample_count')
    weights = counts / counts.sum()
    return (loss * weights).sum(), {key: (value * weights).sum() for key, value in metrics.items()}


def run(args):
    validate_identity(args.pretrained_mode, args.method, args.size)
    torch.set_num_threads(4)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    distributed = world > 1
    if distributed:
        if args.multi_gpu:
            raise ValueError('Use DDP or --multi-gpu DataParallel, not both')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'
        dist.init_process_group('nccl')
    if args.batch_size % world:
        raise ValueError('Global batch size must be divisible by WORLD_SIZE')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    saved = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    previous_init = saved.get('initialization', 'official') if saved else None
    args.init = args.init or previous_init or 'official'
    if saved:
        for key, value in [('pretrained_mode', args.pretrained_mode), ('method', args.method),
                           ('size', args.size), ('input_mode', args.input_mode),
                           ('initialization', args.init)]:
            actual = saved.get(key, 'official' if key == 'initialization' else None)
            if key in ('method', 'size'):
                from encoder.catalog import resolve_identity
                if resolve_identity(args.pretrained_mode, saved['method'], saved['size']) == resolve_identity(args.pretrained_mode, args.method, args.size):
                    continue
            if actual != value:
                raise ValueError(f'Resume checkpoint {key} mismatch: {actual} != {value}')
        if 'training_model' not in saved:
            raise ValueError('Resume requires last.pth with the full training state')
        previous_config = saved.get('training_config', {})
        for key in ('seed', 'samples', 'data_root', 'world_size'):
            expected = world if key == 'world_size' else str(args.data_root) if key == 'data_root' else getattr(args, key)
            if key in previous_config and previous_config[key] != expected:
                raise ValueError(f'Resume requires the original {key}')
        if args.input_mode in DEPTH_DEFORM_SPECS and saved.get('depth_deform_encoding') != depth_deform_spec(args.input_mode):
            raise ValueError('Resume representation version mismatch')
    encoder = ExternalEncoder(args.pretrained_mode, args.method, args.size, metadata=saved,
                              pretrained=saved is None and args.init in ('official', 'adapted'), with_decoder=True)
    encoder.metadata['initialization'] = args.init
    out = args.output_dir or (args.resume.parent if args.resume else None) or result_root(args.input_mode, args.pretrained_mode, args.method, args.size, args.init) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if distributed:
        paths = [str(out)]
        dist.broadcast_object_list(paths, src=0, device=torch.device(args.device))
        out = Path(paths[0])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    metadata = dict(encoder.metadata, input_mode=args.input_mode, format_version=3)
    if args.input_mode in DEPTH_DEFORM_SPECS:
        metadata['depth_deform_encoding'] = depth_deform_spec(args.input_mode)
    # Official exports upstream weights; adapted fine-tunes them on UniVTAC.
    export = args.export_only or (args.init == 'official' and not saved)
    if export:
        if args.init != 'official' or saved:
            raise ValueError('--export-only requires official initialization without --resume')
        if rank == 0:
            save_checkpoint(dict(metadata, encoder=encoder.state_dict()), out / 'encoder.pth')
            (out / 'config.json').write_text(json.dumps(metadata, indent=2))
            print(f'Exported trainable official initializer: {out / "encoder.pth"}', flush=True)
        if distributed:
            dist.destroy_process_group()
        return out
    files = sorted(args.data_root.glob('*/hdf5/*.hdf5'))
    if len(files) < 2:
        raise ValueError('Need at least two contact trajectories for file-level train/validation split')
    random.Random(args.seed).shuffle(files)
    nval = max(1, min(len(files)-1, round(len(files)*.2)))
    val_files, train_files = files[:nval], files[nval:]
    if saved and previous_config.get('train_files') != list(map(str, train_files)):
        raise ValueError('Resume contact file split changed')
    frames = encoder.num_frames
    args.tactile_preprocessing = resolve_tactile_preprocessing(
        args.tactile_preprocessing, args.input_mode, args.device)
    use_cuda_tps = args.tactile_preprocessing == 'cuda'
    offset = encoder.metadata.get('background_offset') if args.pretrained_mode == 'tactile' else None
    train_ds = ExternalFrames(train_files, round(args.samples*.8), seed=args.seed,
                              input_mode=args.input_mode, num_frames=frames, background_offset=offset,
                              tactile_preprocessing=args.tactile_preprocessing)
    val_ds = ExternalFrames(val_files, max(1, round(args.samples*.2)), seed=args.seed+1,
                            input_mode=args.input_mode, num_frames=frames, background_offset=offset,
                            tactile_preprocessing=args.tactile_preprocessing)
    options = dict(batch_size=args.batch_size//world, num_workers=args.workers, pin_memory=str(args.device).startswith('cuda'))
    if use_cuda_tps:
        options.update(collate_fn=collate_tactile_tps, worker_init_fn=initialize_data_worker)
    sampler = DistributedSampler(train_ds, seed=args.seed, drop_last=True) if distributed else None
    train_loader = DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed+rank), **options)
    # No padded/repeated validation anchors. Evaluate the unwrapped model so
    # unequal numbers of batches across ranks cannot deadlock DDP broadcasts.
    val_subset = Subset(val_ds, range(rank, len(val_ds), world)) if distributed else val_ds
    val_loader = DataLoader(val_subset, shuffle=False, **options)
    if not len(train_loader):
        raise ValueError('Encoder training requires at least one global batch of 64 training anchors')
    sparsh = args.pretrained_mode == 'tactile' and args.method not in ('AnyTouch2', 'Anytouch2')
    if sparsh:
        from encoder.sparsh_training import SparshTraining
        core = SparshTraining(encoder, args.mask_ratio)
        objective = core.recipe
    elif encoder.method == 'AnyTouch2':
        from encoder.Anytouch2.training import VideoMAETraining
        core = VideoMAETraining(encoder, args.mask_ratio)
        objective = core.recipe
    else:
        core = VisionAdaptation(encoder, args.mask_ratio, args.kl_weight)
        objective = 'reconstruction_kl' if args.method in ('Wan2.2-VAE', 'Wan22_VAE') else 'masked_latent_and_patch_rgb'
    if encoder.method == 'AnyTouch2':
        metadata.update(training_objective=objective, mae_mask_ratio=args.mask_ratio,
                        mae_target='normalized_tubelet_pixels', mae_decoder_init='random')
    core = core.to(args.device)
    model = core
    if distributed:
        model = nn.parallel.DistributedDataParallel(core, device_ids=[local_rank], broadcast_buffers=False)
    elif args.multi_gpu:
        if not str(args.device).startswith('cuda') or torch.cuda.device_count() < 2:
            raise ValueError('--multi-gpu requires at least two visible CUDA devices')
        if sparsh:
            raise ValueError('Use NUM_PROCESSES for Sparsh multi-GPU training (EMA and centers require DDP)')
        model = nn.DataParallel(core)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(metadata, train_files=list(map(str, train_files)), val_files=list(map(str, val_files)),
                  split_strategy='file', objective=objective, world_size=world,
                  train_count=len(train_ds), val_count=len(val_ds))
    if rank == 0:
        (out / 'config.json').write_text(json.dumps(config, indent=2))
    best, stale, step, start_epoch = float('inf'), 0, 0, 0
    if saved:
        core.load_state_dict(saved['training_model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        best, stale, step, start_epoch = saved['best_val_loss'], saved.get('stale_epochs', 0), saved['step'], saved['epoch']
        rng = saved.get('rank_rng_states', [saved])[rank] if distributed else saved
        if 'rng_state' in rng:
            torch.set_rng_state(rng['rng_state'])
            train_loader.generator.set_state(rng['loader_rng_state'])
            random.setstate(rng.get('python_rng_state', random.getstate()))
            if str(args.device).startswith('cuda'):
                torch.cuda.set_rng_state_all(rng['cuda_rng_states'])
    else:
        # Same parameters, independent stochastic masks on each rank.
        torch.manual_seed(args.seed+rank)
        random.seed(args.seed+rank)
    epoch_steps = min(len(train_loader), args.max_steps or len(train_loader))
    total_steps = epoch_steps * args.epochs
    warmup = min(epoch_steps, max(1, total_steps//10))

    def log(record):
        if rank == 0:
            with (out / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        previous_end = time.monotonic()
        for i, images in enumerate(train_loader):
            if args.max_steps and i >= args.max_steps:
                break
            started = time.monotonic()
            data_wait = started - previous_end
            images = prepare_encoder_batch(images, args.device, args.input_mode)
            factor = (step+1)/warmup if step < warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = args.lr*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(args.device).startswith('cuda')):
                loss, metrics = model(images)
                loss, metrics = reduce_replica_loss(loss, metrics)
            finite = torch.isfinite(loss).int()
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError(f'Nonfinite pretraining loss: {loss}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            core.update_teacher()
            step += 1
            numbers = torch.stack([loss.detach().float()] + [v.detach().float() for v in metrics.values()])
            if distributed:
                dist.all_reduce(numbers)
                numbers /= world
            wait_max = loss.new_tensor(data_wait)
            if distributed:
                dist.all_reduce(wait_max, op=dist.ReduceOp.MAX)
            values, max_wait = numbers.tolist(), float(wait_max)
            elapsed = time.monotonic()-started
            log(dict(epoch=epoch+1, step=step, loss=values[0], lr=optimizer.param_groups[0]['lr'],
                     batch_size=args.batch_size, gpu_count=world if distributed else torch.cuda.device_count() if args.multi_gpu else 1,
                     seconds=elapsed, data_wait_seconds=data_wait,
                     data_wait_max_seconds=max_wait, batch_seconds=elapsed+data_wait,
                     **{key: value for key, value in zip(metrics, values[1:])}))
            previous_end = time.monotonic()
        core.eval()
        seen, total = 0, 0.
        rng_devices = (list(range(torch.cuda.device_count())) if args.multi_gpu else
                       [torch.device(args.device).index or 0]) if str(args.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            torch.manual_seed(args.seed+100000+rank)
            for i, images in enumerate(val_loader):
                if args.max_val_steps and i >= args.max_val_steps:
                    break
                images = prepare_encoder_batch(images, args.device, args.input_mode)
                loss, metrics = core(images)
                loss, _ = reduce_replica_loss(loss, metrics)
                total += float(loss)*len(images)
                seen += len(images)
        aggregate = torch.tensor([total, seen], dtype=torch.float64, device=args.device)
        if distributed:
            dist.all_reduce(aggregate)
        value = float(aggregate[0]/aggregate[1])
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite validation loss')
        improved = value < best
        best, stale = min(best, value), 0 if improved else stale+1
        if args.method in ('Wan2.2-VAE', 'Wan22_VAE') and args.init == 'scratch':
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(args.seed+200000+rank)
                calibration = calibrate_wan(encoder, train_loader, args.device, args.calibration_steps, args.input_mode)
            metadata.update(model_config=encoder.metadata['model_config'], latent_calibration=calibration)
            config.update(model_config=metadata['model_config'], latent_calibration=calibration)
            if rank == 0:
                (out / 'config.json').write_text(json.dumps(config, indent=2))
        state = dict(metadata, encoder=encoder.state_dict(), epoch=epoch+1, step=step, val_loss=value, best_val_loss=best)
        rng = dict(rng_state=torch.get_rng_state(), loader_rng_state=train_loader.generator.get_state(),
                   python_rng_state=random.getstate(),
                   cuda_rng_states=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else [])
        if distributed and not args.no_training_state:
            rank_states = [None]*world
            dist.all_gather_object(rank_states, rng)
        if rank == 0:
            if improved:
                save_checkpoint(state, out / 'encoder.pth')
                if args.method in ('Wan2.2-VAE', 'Wan22_VAE'):
                    save_checkpoint(dict(state, training_model=core.state_dict()), out / 'best.pth')
            if not args.no_training_state:
                save_checkpoint(dict(state, training_model=core.state_dict(), optimizer=optimizer.state_dict(),
                                training_config=config, stale_epochs=stale, **rng,
                                **({'rank_rng_states': rank_states} if distributed else {})), out / 'last.pth')
        log(dict(epoch=epoch+1, step=step, val_loss=value, best_val_loss=best, best=improved))
        if args.early_stop_patience and stale >= args.early_stop_patience:
            break
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return out


def main(default_method=None, default_mode=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrained-mode', required=default_mode is None, default=default_mode, choices=['tactile', 'vision'])
    p.add_argument('--method', required=default_method is None, default=default_method)
    p.add_argument('--size', default='official', choices=['official'])
    p.add_argument('--init', choices=['scratch', 'official', 'adapted'],
                   help='scratch: random pretraining; official: export upstream weights; adapted: fine-tune upstream weights on UniVTAC')
    p.add_argument('--export-only', action='store_true', help='Export official weights without pretraining')
    p.add_argument('--input-mode', choices=INPUT_CHANNELS, default='depth_deform_nonlinear_radial')
    p.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int)
    p.add_argument('--samples', type=int, default=100000)
    p.add_argument('--batch-size', type=int, choices=[64])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tactile-preprocessing', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='Use batched GPU TPS for nonlinear inputs on CUDA; cpu retains the reference')
    p.add_argument('--lr', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--mask-ratio', type=float)
    p.add_argument('--kl-weight', type=float)
    p.add_argument('--calibration-steps', type=int,
                   help='Training batches for scratch Wan latent mean/std per epoch')
    p.add_argument('--max-steps', type=int)
    p.add_argument('--max-val-steps', type=int)
    p.add_argument('--early-stop-patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--multi-gpu', action=argparse.BooleanOptionalAction, default=None,
                   help='Split the fixed global batch of 64 across visible CUDA devices')
    p.add_argument('--no-training-state', action='store_true')
    args = p.parse_args()
    from encoder.catalog import selection
    args.pretrained_mode, args.method, args.size, _ = selection(args.pretrained_mode, args.method, args.size, args.init)
    defaults = dict(epochs=5, lr=1e-5, weight_decay=.05, batch_size=64, multi_gpu=False)
    resume_config = torch.load(args.resume, map_location='cpu', weights_only=True).get('training_config', {}) if args.resume else {}
    config_path = args.config or ROOT / 'encoder' / args.method / 'config.yml'
    if config_path.exists():
        settings = yaml.safe_load(config_path.read_text())
        defaults.update(settings.get(args.method, settings))
    if args.resume:
        defaults.update({key: resume_config[key] for key in defaults if key in resume_config})
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.batch_size != 64:
        p.error('All encoder training batches must be 64, including config/resume overrides')
    if args.epochs < 1 or args.batch_size < 1 or args.samples < 2:
        p.error('epochs/batch size must be positive and samples must be >=2')
    if args.mask_ratio is None:
        args.mask_ratio = resume_config.get('mask_ratio', .75 if args.pretrained_mode == 'tactile' and args.method in ('MAE', 'Sparsh_mae') else .5)
    if args.kl_weight is None:
        args.kl_weight = resume_config.get('kl_weight', 1e-6)
    if args.calibration_steps is None:
        args.calibration_steps = resume_config.get('calibration_steps', 16)
    if not 0 < args.mask_ratio < 1 or args.calibration_steps < 1:
        p.error('mask ratio must be in (0,1), calibration steps must be positive')
    if any(v is not None and v < 1 for v in (args.max_steps, args.max_val_steps)):
        p.error('step limits must be positive')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        args.multi_gpu = False
    run(args)


if __name__ == '__main__':
    main()
