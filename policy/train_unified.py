"""Train ACT, PI0.5, or StarVLA-GR00T on UniVTAC raw HDF5 demonstrations."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import random
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "encoder"))
from modality import DEPTH_DEFORM_SPECS, depth_deform_spec, INPUT_CHANNELS, read_tactile  # noqa: E402
from encoder.image_io import read_rgb, require_rgb_model
from policy.joint_actions import encode_joint_actions, joint_action_mode  # noqa: E402


class TeeStream:
    """Keep terminal output while recording it in this rank's result directory."""

    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def training_log(args):
    output = args.output_dir
    error = None
    if args.rank == 0:
        try:
            if any((output / name).exists() for name in
                   ("train_config.yml", "dataset_stats.json", "episodes.json")) or any(output.glob("checkpoint_*.pt")):
                raise FileExistsError(f"Training output already contains a run: {output}")
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error = str(exc)
    if args.world_size > 1:
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error:
        raise FileExistsError(error)
    log_path = output / ("train.log" if args.rank == 0 else f"train_rank{args.rank}.log")
    stdout, stderr = sys.stdout, sys.stderr
    status = 1
    with log_path.open("a", buffering=1) as logfile:
        sys.stdout, sys.stderr = TeeStream(stdout, logfile), TeeStream(stderr, logfile)
        try:
            print(f"Training log: {log_path}", flush=True)
            yield
            status = 0
        except BaseException:
            traceback.print_exc(file=logfile)
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            if args.rank == 0:
                (output / "exit_code").write_text(f"{status}\n")


class DistributedWeightedSampler(Sampler):
    """Shard one deterministic, task-balanced draw across DDP ranks."""

    def __init__(self, weights, num_replicas, rank, seed):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = (len(weights) + num_replicas - 1) // num_replicas

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.num_samples * self.num_replicas,
                                    replacement=True, generator=generator)
        return iter(indices[self.rank::self.num_replicas].tolist())


def setup_distributed(args):
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.batch_size % args.world_size:
        raise ValueError(f"Global batch size {args.batch_size} must be divisible by world size {args.world_size}")
    args.batch_size_per_rank = args.batch_size // args.world_size
    if args.world_size > 1:
        if str(args.device).startswith("cuda"):
            torch.cuda.set_device(args.local_rank)
            args.device = f"cuda:{args.local_rank}"
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)


def episode_files(task: str, limit: int) -> list[Path]:
    folder = ROOT / "resources" / "data" / "isaac51" / task / "hdf5"
    files = sorted(folder.glob("*.hdf5"), key=lambda p: int(p.stem))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes in {folder}")
    return files


def split_files(tasks: list[str], limit: int, seed: int) -> tuple[dict, dict]:
    train, val = {}, {}
    for task in tasks:
        files = episode_files(task, limit)
        shuffled = files.copy()
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, round(len(files) * 0.1)) if len(files) > 1 else 0
        val[task], train[task] = shuffled[:n_val], shuffled[n_val:]
    return train, val


def relative_action_stats(files_by_task, chunk_size, sample_limit=200_000):
    """Exact moments over valid chunk targets; bounded uniform sample for quantiles."""
    rng = np.random.default_rng(42)
    count = 0
    total, squares = np.zeros(8), np.zeros(8)
    samples, priorities = np.empty((0, 8), np.float32), np.empty(0)
    threshold = 1.0
    for files in files_by_task.values():
        for path in files:
            with h5py.File(path, "r") as f:
                joint = np.asarray(f["embodiment/joint"][:, :8], dtype=np.float32)
            for offset in range(1, min(chunk_size, len(joint) - 1) + 1):
                targets = encode_joint_actions(joint[:-offset], joint[offset:, None, :], "relative_joint")[:, 0]
                count += len(targets)
                total += targets.sum(0, dtype=np.float64)
                squares += np.square(targets.astype(np.float64)).sum(0)
                keys = rng.random(len(targets))
                keep = keys < threshold
                samples = np.concatenate((samples, targets[keep]))
                priorities = np.concatenate((priorities, keys[keep]))
                if len(samples) > 2 * sample_limit:
                    keep = np.argpartition(priorities, sample_limit - 1)[:sample_limit]
                    samples, priorities = samples[keep], priorities[keep]
                    threshold = priorities.max()
    if not count:
        raise ValueError("No trajectory has at least two joint frames")
    if len(samples) > sample_limit:
        keep = np.argpartition(priorities, sample_limit - 1)[:sample_limit]
        samples = samples[keep]
    mean = total / count
    return {"mean": mean.tolist(), "std": np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), 1e-2).tolist(),
            "q01": np.quantile(samples, 0.01, axis=0).tolist(),
            "q99": np.quantile(samples, 0.99, axis=0).tolist()}


def compute_stats(files_by_task: dict[str, list[Path]], action_mode="absolute_joint", chunk_size=1) -> dict:
    action_mode = joint_action_mode(action_mode)
    states, actions = [], []
    for files in files_by_task.values():
        for path in files:
            with h5py.File(path, "r") as f:
                joint = np.asarray(f["embodiment/joint"][:, :8], dtype=np.float32)
            if len(joint) < 2:
                continue
            states.append(joint[:-1])
            actions.append(joint[1:])
    if not states:
        raise ValueError("No trajectory has at least two joint frames")
    state, action = np.concatenate(states), np.concatenate(actions)
    result = {}
    for name, array in (("state", state), ("action", action)):
        result[name] = {
            "mean": array.mean(0).tolist(),
            "std": np.maximum(array.std(0), 1e-2).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }
    if action_mode == "relative_joint":
        result["action"] = relative_action_stats(files_by_task, chunk_size)
    result["action_space"] = {"mode": action_mode, "chunk_size": chunk_size,
                              "anchor": "current_observation", "gripper": "absolute"}
    return result


class UniVTACDataset(Dataset):
    def __init__(self, files_by_task, *, cameras, tactile_mode, tactile_input_mode,
                 chunk_size, stats, policy, image_size, action_mode="absolute_joint",
                 tactile_num_frames=1, tactile_frame_stride=1, tactile_pretrained_mode='univtac',
                 tactile_background_offset=.5, tactile_preprocessing='cpu', max_open_files=32):
        if not isinstance(max_open_files, int) or max_open_files < 1:
            raise ValueError('max_open_files must be a positive integer')
        self.rows = []
        self.task_counts = defaultdict(int)
        self.cameras = cameras
        self.tactile_mode = tactile_mode
        self.tactile_input_mode = tactile_input_mode
        self.chunk_size = chunk_size
        self.stats = stats
        self.policy = policy
        self.image_size = image_size
        self.action_mode = joint_action_mode(action_mode)
        self._baselines = {}
        self.tactile_num_frames = tactile_num_frames
        self.tactile_frame_stride = tactile_frame_stride
        self.tactile_pretrained_mode = tactile_pretrained_mode
        self.tactile_background_offset = tactile_background_offset
        self.tactile_preprocessing = tactile_preprocessing
        # A persistent worker visits the whole shuffled dataset. Retaining every
        # HDF5 handle also retains each file's internal caches, multiplied by
        # all workers on every DDP rank. Bound handles and their RGB baselines.
        self.max_open_files = max_open_files
        self._files = OrderedDict()
        for task, paths in files_by_task.items():
            for path in paths:
                with h5py.File(path, "r") as f:
                    length = len(f["embodiment/joint"])
                    for camera in cameras:
                        if f"observation/{camera}/rgb" not in f:
                            raise KeyError(f"{path}: missing {camera} camera")
                    if tactile_mode != "none":
                        for side in ("left", "right"):
                            if f"tactile/{side}_tactile" not in f:
                                raise KeyError(f"{path}: missing {side} tactile stream")
                for t in range(max(0, length - 1)):
                    self.rows.append((task, path, t))
                    self.task_counts[task] += 1

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()
        state["_baselines"] = {}
        return state

    def _file(self, path):
        handle = self._files.pop(path, None)
        if handle is None:
            if len(self._files) >= self.max_open_files:
                old_path, old_handle = self._files.popitem(last=False)
                old_handle.close()
                for side in ('left', 'right'):
                    self._baselines.pop((old_path, side), None)
            handle = h5py.File(path, "r")
        self._files[path] = handle
        return handle

    def _norm(self, x, name):
        s = self.stats[name]
        if self.policy in ("pi05", "n0vtla", "starvla_pi"):
            low = np.asarray(s["q01"], dtype=np.float32)
            high = np.asarray(s["q99"], dtype=np.float32)
            return 2 * (x - low) / np.maximum(high - low, 1e-6) - 1
        return (x - np.asarray(s["mean"], dtype=np.float32)) / np.asarray(s["std"], dtype=np.float32)

    def _rgb(self, f, key, t):
        image = read_rgb(f[key], t)
        if self.policy == "n0vtla":
            from policy.n0vtla.observation_adapter import resize_rgb
            return resize_rgb(image)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255

    def __getitem__(self, index):
        task, path, t = self.rows[index]
        f = self._file(path)
        joint = np.asarray(f["embodiment/joint"][t:min(t + self.chunk_size + 1, len(f["embodiment/joint"])), :8], dtype=np.float32)
        state = joint[0]
        action = joint[1:]
        valid = len(action)
        if valid < self.chunk_size:
            action = np.concatenate((action, np.repeat(action[-1:], self.chunk_size - valid, axis=0)))
        action = encode_joint_actions(state, action, self.action_mode)
        out = {
            "observation.state": torch.from_numpy(self._norm(state, "state")),
            "action": torch.from_numpy(self._norm(action, "action")),
            "is_pad": torch.arange(self.chunk_size) >= valid,
            "task": task.replace("_", " "),
        }
        for camera in self.cameras:
            key = "cam_high" if camera == "head" else "cam_wrist"
            out[f"observation.images.{key}"] = self._rgb(f, f"observation/{camera}/rgb", t)
        if self.tactile_mode != "none":
            for side in ("left", "right"):
                if self.tactile_mode == "native":
                    from policy.n0vtla.observation_adapter import read_native_tactile
                    tactile = read_native_tactile(f, side, t, self.tactile_input_mode)
                else:
                    from encoder.pretrained_registry import history_indices
                    indices = history_indices(t, self.tactile_num_frames, self.tactile_frame_stride)
                    if self.tactile_preprocessing == 'cuda':
                        from encoder.tactile_preprocessing import prepare_tactile_tps
                        out[f"observation.images.tac_{side}"] = [
                            prepare_tactile_tps(f, side, i, self.image_size) for i in indices]
                        continue
                    frames = [read_tactile(f, side, i, self.tactile_input_mode, self.image_size) for i in indices]
                    tactile = torch.stack(frames) if self.tactile_num_frames > 1 else frames[0]
                    if self.tactile_pretrained_mode == 'tactile' and self.tactile_input_mode in ('rgb_only', 'marker_rgb'):
                        key = (path, side)
                        if key not in self._baselines:
                            self._baselines[key] = read_tactile(f, side, 0, self.tactile_input_mode, self.image_size)
                        tactile = (tactile - self._baselines[key] + self.tactile_background_offset).clamp(0, 1)
                if self.tactile_mode == "as_image" and tactile.shape[0] == 1:
                    tactile = tactile.repeat(3, 1, 1)
                out[f"observation.images.tac_{side}"] = tactile
                if self.tactile_mode == "native":
                    key = (path, side)
                    if key not in self._baselines:
                        self._baselines[key] = read_native_tactile(f, side, 0, self.tactile_input_mode)
                    out[f"observation.images.tac_{side}.baseline"] = self._baselines[key]
        return out


def select_tasks(args):
    settings = json.loads((ROOT / "policy" / "task_settings.json").read_text())
    available = [name for name in settings if (ROOT / "resources" / "data" / "isaac51" / name / "hdf5").is_dir()]
    if args.training_mode == "single_task":
        if args.task not in available:
            raise ValueError(f"Unknown or unavailable task {args.task}; available: {available}")
        return [args.task]
    if args.tasks:
        tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
        unknown = set(tasks) - set(available)
        if unknown:
            raise ValueError(f"Unavailable tasks: {sorted(unknown)}")
        return tasks
    return available


def make_config(args, base, camera_keys, tactile_keys):
    base = dict(base, chunk_size=args.chunk_size, n_action_steps=args.n_action_steps)
    if args.policy == "n0vtla":
        from policy.n0vtla.configuration_n0vtla import N0PolicyConfig
        return N0PolicyConfig(**{**base, "chunk_size": args.chunk_size,
                               "n_action_steps": args.n_action_steps})
    from policy.vla_common.engine.configs import FeatureType, PolicyFeature
    features = {key: PolicyFeature(FeatureType.VISUAL, (3, args.image_size, args.image_size)) for key in camera_keys}
    features["observation.state"] = PolicyFeature(FeatureType.STATE, (8,))
    if args.tactile_mode != "none":
        channels = 3 if args.tactile_mode == "as_image" else INPUT_CHANNELS[args.tactile_input_mode]
        features.update({key: PolicyFeature(FeatureType.VISUAL, (channels, args.image_size, args.image_size)) for key in tactile_keys})
    common = dict(base, input_features=features,
                  output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
                  top_camera_keys=[key for key in camera_keys if key.endswith("cam_high")],
                  wrist_camera_keys=[key for key in camera_keys if key.endswith("cam_wrist")],
                  tactile_keys=tactile_keys, tactile_mode=args.tactile_mode,
                  tactile_input_mode=args.tactile_input_mode, tactile_type=args.tactile_type,
                  tactile_encoder_pretrained_mode=args.encoder_pretrained_mode,
                  tactile_num_frames=args.tactile_num_frames, tactile_frame_offset=args.tactile_frame_stride,
                  freeze_tactile_encoder=False,
                  wan_training_optimization=getattr(args, 'wan_training_optimization', 'compiled'),
                  tactile_encoder_path=str(args.encoder_ckpt) if args.tactile_mode == "encode" else None,
                  tactile_insert_location=args.tactile_insert_location,
                  state_mode="absolute_joint", action_mode=args.action_mode, action_gap=1,
                  device=args.device, push_to_hub=False)
    common["chunk_size"] = args.chunk_size
    if args.policy == "pi05":
        from policy.pi05.configuration_pi05 import PI05Config
        common["paligemma_tokenizer_path"] = str(ROOT / "resources" / "pretrained_models" / "pi05_base" / "paligemma-3b-pt-224-tokenizer")
        return PI05Config(**common)
    from policy.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
    common["base_vlm"] = str(ROOT / "resources" / "pretrained_models" / "Qwen3.5-2B")
    common["action_dim"] = 8
    common["state_dim"] = 8
    if args.policy == "starvla_pi":
        from policy.starvla_pi import StarvlaPiConfig
        return StarvlaPiConfig(**common)
    return StarvlaGrootConfig(**common)


from policy.pi05.tokenization import make_pi_tokens


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.tactile_mode == 'encode':
        from encoder.catalog import selection
        args.encoder_pretrained_mode, args.encoder_method, args.encoder_size, _ = selection(
            args.encoder_pretrained_mode, args.encoder_method, args.encoder_size,
            getattr(args, 'encoder_pretrain_init', None))
    args.tactile_num_frames, args.tactile_frame_stride = 1, 1
    if args.tactile_mode == 'encode':
        from encoder.pretrained_registry import validate_identity
        validate_identity(args.encoder_pretrained_mode, args.encoder_method, args.encoder_size)
        requested_init = getattr(args, 'encoder_pretrain_init', None)
        if not args.encoder_ckpt and requested_init in ('scratch', 'adapted'):
            raise ValueError(f'{requested_init} requires an explicit trained encoder checkpoint')
        if not args.encoder_ckpt and (args.encoder_pretrained_mode == 'tactile' or
                                     (args.encoder_pretrained_mode == 'vision' and args.encoder_method in ('Wan2.2-VAE', 'Wan22_VAE'))):
            # Export once on rank zero. No download or random fallback during policy restore.
            paths = [None]
            if args.rank == 0:
                from encoder.external import initial_metadata
                paths[0] = str(initial_metadata(args.encoder_pretrained_mode, args.encoder_method, args.encoder_size, args.tactile_input_mode))
            if args.world_size > 1:
                dist.broadcast_object_list(paths, src=0)
            args.encoder_ckpt = Path(paths[0])
        if not args.encoder_ckpt:
            raise ValueError('encode mode requires a trained encoder checkpoint')
        encoder_state = torch.load(args.encoder_ckpt, map_location='cpu', weights_only=True)
        from encoder.pretrained_registry import require_initialization
        args.encoder_pretrain_init = require_initialization(encoder_state, requested_init)
        if encoder_state.get('pretrained_mode', 'univtac') != args.encoder_pretrained_mode:
            raise ValueError('Encoder checkpoint pretrained_mode does not match requested mode')
        from encoder.catalog import resolve_identity
        actual_identity = resolve_identity(args.encoder_pretrained_mode,
            encoder_state.get('method', args.encoder_method), encoder_state.get('size', args.encoder_size))
        expected_identity = resolve_identity(args.encoder_pretrained_mode, args.encoder_method, args.encoder_size)
        if actual_identity != expected_identity:
            raise ValueError('Encoder checkpoint method/size does not match requested model')
        args.tactile_num_frames = encoder_state.get('num_frames', 1)
        args.tactile_frame_stride = encoder_state.get('frame_stride', 1)
    if args.tactile_mode == "encode" and args.tactile_input_mode in ("rgb_only", "marker_rgb"):
        require_rgb_model(torch.load(args.encoder_ckpt, map_location="cpu", weights_only=True))
    if args.tactile_mode == "encode" and args.tactile_input_mode in DEPTH_DEFORM_SPECS:
        encoder_state = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
        if encoder_state.get("depth_deform_encoding") != depth_deform_spec(args.tactile_input_mode):
            raise ValueError("depth_deform encoder uses a different normalization; retrain it with the selected depth/deform ranges before policy training")
    if args.policy == "n0vtla":
        if args.tactile_mode not in ("native", "none"):
            raise ValueError("n0vtla supports tactile-mode native or none")
        if args.image_size != 224:
            raise ValueError("N0 requires image-size=224")
    elif args.tactile_mode == "native":
        raise ValueError("native tactile mode requires policy=n0vtla")
    tasks = select_tasks(args)
    train_files, val_files = split_files(tasks, args.episodes, args.seed)
    if args.world_size > 1:
        stats_holder = [compute_stats(train_files, args.action_mode, args.chunk_size) if args.rank == 0 else None]
        dist.broadcast_object_list(stats_holder, src=0)
        stats = stats_holder[0]
    else:
        stats = compute_stats(train_files, args.action_mode, args.chunk_size)
    settings = json.loads((ROOT / "policy" / "task_settings.json").read_text())
    if args.cameras == "auto":
        cameras = ["head", "wrist"] if all(settings[t].get("camera_type") == "all" for t in tasks) else ["head"]
    else:
        cameras = ["head", "wrist"] if args.cameras == "all" else ["head"]
    image_size = 256 if args.policy == "act" else args.image_size
    tps_mode = args.tactile_input_mode in ('depth_deform_nonlinear', 'depth_deform_nonlinear_radial')
    use_cuda_tps = (args.tactile_preprocessing != 'cpu' and str(args.device).startswith('cuda')
                    and tps_mode and args.tactile_mode in ('encode', 'as_image'))
    if args.tactile_preprocessing == 'cuda' and not use_cuda_tps:
        raise ValueError('CUDA tactile preprocessing requires a nonlinear TPS input, CUDA device, and encode/as_image mode')
    args.tactile_preprocessing = 'cuda' if use_cuda_tps else 'cpu'
    make_ds = lambda files: UniVTACDataset(files, cameras=cameras, tactile_mode=args.tactile_mode,
                                           tactile_input_mode=args.tactile_input_mode,
                                           chunk_size=args.chunk_size, stats=stats,
                                           policy=args.policy, image_size=image_size, action_mode=args.action_mode,
                                           tactile_num_frames=args.tactile_num_frames,
                                           tactile_frame_stride=args.tactile_frame_stride,
                                           tactile_pretrained_mode=args.encoder_pretrained_mode,
                                           tactile_background_offset=130/255 if args.encoder_method in ('AnyTouch2', 'Anytouch2') else .5,
                                           tactile_preprocessing=args.tactile_preprocessing)
    train_ds, val_ds = make_ds(train_files), make_ds(val_files)
    weights = [1.0 / train_ds.task_counts[task] for task, _, _ in train_ds.rows]
    if args.world_size > 1:
        sampler = (DistributedWeightedSampler(weights, args.world_size, args.rank, args.seed)
                   if args.training_mode == "multi_task" else
                   DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank,
                                      seed=args.seed, drop_last=True))
    else:
        sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True) if args.training_mode == "multi_task" else None
    # Spawn avoids forking after CUDA/NCCL has been initialized.
    loader_options = {"multiprocessing_context": "spawn"} if args.workers else {}
    if args.workers and train_ds.tactile_num_frames > 1 and train_ds.tactile_input_mode in ('rgb_only', 'marker_rgb'):
        # Temporal RGB batches contain four frames per sensor. One outstanding
        # batch per worker bounds queued tensors and their pinned-memory copies.
        loader_options["prefetch_factor"] = 1
    if args.workers and args.policy in ("starvla_groot", "starvla_pi"):
        loader_options["persistent_workers"] = True
    if use_cuda_tps:
        from encoder.tactile_preprocessing import collate_tactile_tps, initialize_data_worker
        loader_options.update(collate_fn=collate_tactile_tps, worker_init_fn=initialize_data_worker)
    loader = DataLoader(train_ds, batch_size=args.batch_size_per_rank, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed + args.rank), **loader_options)
    if len(loader) == 0:
        raise ValueError("Training dataset is smaller than one batch")
    # Validation is run by rank 0 on the unwrapped model, at the per-GPU batch size.
    # Load validation synchronously: capped validation exits early, and worker
    # teardown with prefetched batches can abort during iterator cleanup.
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size_per_rank, num_workers=0,
                            **({'collate_fn': collate_tactile_tps} if use_cuda_tps else {}))
                  if len(val_ds) and args.rank == 0 else None)
    output = args.output_dir
    camera_keys = [f"observation.images.cam_{'high' if c == 'head' else 'wrist'}" for c in cameras]
    tactile_keys = [f"observation.images.tac_{side}" for side in ("left", "right")]
    if args.policy == "act":
        sys.path.insert(0, str(ROOT / "policy" / "ACT"))
        from act_policy import ACTPolicy
        base = yaml.safe_load((ROOT / "policy" / "ACT" / "base_config.yml").read_text())
        base.update(chunk_size=args.chunk_size, n_action_steps=args.n_action_steps, camera_names=["cam_high" if c == "head" else "cam_wrist" for c in cameras],
                    tactile_names=["tac_left", "tac_right"] if args.tactile_mode != "none" else [],
                    tactile_ckpt=str(args.encoder_ckpt) if args.tactile_mode == "encode" else None,
                    tactile_input_mode=args.tactile_input_mode, tactile_type=args.tactile_type,
                    tactile_encoder_pretrained_mode=args.encoder_pretrained_mode,
                    wan_training_optimization=getattr(args, 'wan_training_optimization', 'compiled'),
                    batch_size=args.batch_size, num_steps=args.steps, save_freq=args.save_freq, device=args.device,
                    ckpt_dir=str(output), task_name=tasks[0] if len(tasks) == 1 else "multi_task",
                    seed=args.seed, num_epochs=1)
        if args.tactile_mode == "as_image":
            base["tactile_encoder_method"] = "ResNet"
            base["tactile_encoder_size"] = "S"
        else:
            base["tactile_encoder_method"] = args.encoder_method
            base["tactile_encoder_size"] = args.encoder_size
        model = ACTPolicy(base).to(args.device)
        optimizer = model.configure_optimizers()
        scheduler = None
        config_dump = {**base, "policy": args.policy, "training_mode": args.training_mode, "tasks": tasks,
                       "tactile_mode": args.tactile_mode,
                       "routing": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        if args.tactile_mode == "encode":
            encoder_state = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=True)
            from encoder.pretrained_registry import constructor_metadata
            config_dump["tactile_encoder_config"] = constructor_metadata(encoder_state)
        tokenizer = None
    elif args.policy == "n0vtla":
        from transformers import AutoTokenizer
        from policy.n0vtla.modeling_n0vtla import N0Policy
        base = yaml.safe_load((ROOT / "policy/n0vtla/base_config.yml").read_text())
        base.update(chunk_size=args.chunk_size, n_action_steps=args.n_action_steps)
        config = make_config(args, base, camera_keys, tactile_keys)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        model = N0Policy(config)
        model.load_base(args.pretrained_path)
        model.to(args.device)
        optim_params = model.get_optim_params()
        optimizer = torch.optim.AdamW(model.optimizer_groups(), lr=config.optimizer_lr,
            betas=tuple(config.optimizer_betas), eps=config.optimizer_eps, weight_decay=config.optimizer_weight_decay)
        scheduler = config.get_scheduler_preset().build(optimizer, args.steps)
        config_dump = {"policy": args.policy, "n0_config": asdict(config),
                       "attention_backend": "eager",
                       "native_model": asdict(config.native_config()), "loss_mask": "valid_steps_first_8_dims",
                       "camera_names": [key.split(".")[-1] for key in camera_keys],
                       "chunk_size": args.chunk_size,
                       "routing": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    else:
        base = yaml.safe_load((ROOT / "policy" / args.policy / "base_config.yml").read_text())
        base.update(chunk_size=args.chunk_size, n_action_steps=args.n_action_steps)
        config = make_config(args, base, camera_keys, tactile_keys)
        config.validate_features()
        if args.policy == "pi05":
            from transformers import AutoTokenizer
            from policy.pi05.modeling_pi05 import PI05Policy
            tokenizer = AutoTokenizer.from_pretrained(config.paligemma_tokenizer_path, local_files_only=True)
            model = PI05Policy.from_pretrained(ROOT / "resources" / "pretrained_models" / "pi05_base", config=config, local_files_only=True)
        else:
            from policy.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy
            tokenizer = None
            if args.policy == "starvla_pi":
                from policy.starvla_pi.modeling_starvla_pi import StarvlaPiPolicy
                model = StarvlaPiPolicy(config)
            else:
                model = StarvlaGrootPolicy(config)
        model.to(args.device)
        # Keep update/clipping order stable when a policy orders its modules
        # differently to overlap DDP communication with backward computation.
        optim_params = list(model.get_optim_params())
        optimizer = torch.optim.AdamW(optim_params, lr=config.optimizer_lr,
                                      betas=tuple(config.optimizer_betas), eps=config.optimizer_eps,
                                      weight_decay=config.optimizer_weight_decay,
                                      fused=(str(args.device).startswith("cuda") and
                                             args.policy in ("starvla_groot", "starvla_pi")))
        scheduler = config.get_scheduler_preset().build(optimizer, args.steps)
        config_dump = {"policy": args.policy, "base": base,
                       "chunk_size": config.chunk_size,
                       "routing": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       "state_mode": "absolute_joint", "action_mode": args.action_mode, "action_gap": 1,
                       "tactile_encoder_config": config.tactile_encoder_config,
                       "input_features": {k: list(v.shape) for k, v in config.input_features.items()},
                       "output_features": {k: list(v.shape) for k, v in config.output_features.items()}}
    if args.tactile_input_mode in DEPTH_DEFORM_SPECS:
        config_dump["depth_deform_encoding"] = depth_deform_spec(args.tactile_input_mode)
    config_dump["color_order"] = "RGB"
    config_dump.update(tactile_num_frames=args.tactile_num_frames,
                       tactile_frame_stride=args.tactile_frame_stride,
                       encoder_pretrained_mode=args.encoder_pretrained_mode)
    config_dump.update(state_mode="absolute_joint", action_mode=args.action_mode, action_gap=1,
                       n_action_steps=args.n_action_steps)
    config_dump["distributed"] = {"world_size": args.world_size, "global_batch_size": args.batch_size,
                                   "batch_size_per_rank": args.batch_size_per_rank}
    if args.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        from policy.inference_assets import export_assets
        inference = export_assets(output, args.policy,
            qwen=model.qwen_vl if args.policy in ("starvla_groot", "starvla_pi") else None,
            tokenizer=tokenizer)
        if inference is not None:
            config_dump["inference"] = inference
        (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
        (output / "episodes.json").write_text(json.dumps({"train": {k: [p.name for p in v] for k, v in train_files.items()},
                                                            "val": {k: [p.name for p in v] for k, v in val_files.items()}}, indent=2))
        (output / "train_config.yml").write_text(yaml.safe_dump(config_dump, sort_keys=False))
        print(f"Tasks: {tasks}; train rows={len(train_ds)}; val rows={len(val_ds)}; cameras={cameras}; "
              f"action_mode={args.action_mode}; world_size={args.world_size}; global_batch={args.batch_size}; per_gpu_batch={args.batch_size_per_rank}", flush=True)
    core_model = model
    if args.world_size > 1:
        dist.barrier()
        model = DistributedDataParallel(
            core_model,
            device_ids=[args.local_rank] if str(args.device).startswith("cuda") else None,
            find_unused_parameters=True, broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    # Same initialization on every rank, independent dropout/noise during training.
    random.seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    profiler = None
    if args.profile_stages:
        if not str(args.device).startswith('cuda'):
            raise ValueError('Stage profiling requires CUDA')
        from policy.training_profile import TrainingStageProfiler
        profiler = TrainingStageProfiler(core_model, output, args.profile_warmup)

    def batch_loss(raw, forward_model):
        if tokenizer is not None:
            if args.policy == "n0vtla":
                from policy.n0vtla.observation_adapter import tokenize_batch
                tokenize_batch(raw, tokenizer, config.tokenizer_max_length)
            else:
                make_pi_tokens(raw, tokenizer, config.tokenizer_max_length)
        if profiler and profiler.current is not None:
            profiler.mark('h2d_start')
        batch = {k: v.to(args.device, non_blocking=True)
                 if isinstance(v, torch.Tensor) and not k.endswith('.tps_source_hw') else v
                 for k, v in raw.items()}
        if profiler and profiler.current is not None:
            profiler.mark('h2d_end')
        if use_cuda_tps:
            from encoder.tactile_preprocessing import materialize_tactile_tps
            materialize_tactile_tps(batch, args.tactile_input_mode)
        if profiler and profiler.current is not None:
            profiler.mark('tps_end')
        if args.policy == "act":
            cam = torch.stack([batch[k] for k in camera_keys], dim=1)
            mean = torch.tensor([0.485, 0.456, 0.406], device=cam.device)[None, None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=cam.device)[None, None, :, None, None]
            cam = (cam - mean) / std
            tac = torch.stack([batch[k] for k in tactile_keys], dim=1) if args.tactile_mode != "none" else torch.empty(0, device=args.device)
            return forward_model(batch["observation.state"], cam, tac, batch["action"], batch["is_pad"])["loss"]
        if args.policy in ("starvla_groot", "starvla_pi"):
            return forward_model(batch, return_loss_dict=False)[0]
        return forward_model(batch)[0]

    epoch = 0
    train_iter = iter(loader)
    log_start, last_log_step, data_wait_seconds = time.perf_counter(), 0, 0.0
    for step in range(1, args.steps + 1):
        if profiler:
            profiler.begin(step)
        data_start = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_iter = iter(loader)
            batch = next(train_iter)
        step_data_wait = time.perf_counter() - data_start
        data_wait_seconds += step_data_wait
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(batch, model)
        if profiler:
            profiler.mark('forward_end')
        loss.backward()
        if profiler:
            profiler.mark('backward_end')
        if args.policy != "act":
            torch.nn.utils.clip_grad_norm_(optim_params, config.optimizer_grad_clip_norm)
        if profiler:
            profiler.mark('clip_end')
        optimizer.step()
        if profiler:
            profiler.mark('optimizer_end')
        if scheduler:
            scheduler.step()
        if profiler:
            profiler.finish(1000 * step_data_wait)
        if step == 1 or step % args.log_freq == 0:
            count = step - last_log_step
            mean_loss = loss.detach().float().clone()
            max_data_wait = mean_loss.new_tensor(1000 * data_wait_seconds / count)
            if args.world_size > 1:
                dist.all_reduce(mean_loss)
                mean_loss /= args.world_size
                dist.all_reduce(max_data_wait, op=dist.ReduceOp.MAX)
            if args.rank == 0:
                value = float(mean_loss)  # Also waits for this logging window's GPU work.
                elapsed = time.perf_counter() - log_start
                print(f"step={step} loss={value:.6f} lr={optimizer.param_groups[0]['lr']:.3g} "
                      f"step/s={count / elapsed:.3f} data_wait_ms={1000 * data_wait_seconds / count:.1f} "
                      f"data_wait_max_ms={float(max_data_wait):.1f}", flush=True)
            log_start, last_log_step, data_wait_seconds = time.perf_counter(), step, 0.0
        if step % args.save_freq == 0 or step == args.steps:
            if val_loader and args.max_val_batches:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_index, val_batch in enumerate(val_loader):
                        if val_index >= args.max_val_batches:
                            break
                        val_losses.append(float(batch_loss(val_batch, core_model).detach()))
                print(f"step={step} val_loss={np.mean(val_losses):.6f}", flush=True)
            if args.save_checkpoint and args.rank == 0:
                torch.save({"format_version": 2, "color_order": "RGB", "policy": args.policy,
                            "action_mode": args.action_mode,
                            "model": core_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "step": step}, output / f"checkpoint_{step}.pt")
            if args.world_size > 1:
                dist.barrier()
            # Report training throughput separately from validation/checkpoint I/O.
            log_start, last_log_step, data_wait_seconds = time.perf_counter(), step, 0.0
    if profiler:
        profiler.save()
    if args.rank == 0:
        print(f"Finished: {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["act", "pi05", "starvla_groot", "starvla_pi", "n0vtla"], required=True)
    p.add_argument("--pretrained-path", type=Path, default=ROOT / "resources/pretrained_models/n0-vtla-base")
    p.add_argument("--tokenizer-path", type=Path,
                   default=ROOT / "resources/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer")
    p.add_argument("--training-mode", choices=["single_task", "multi_task"], default="single_task")
    p.add_argument("--task", default="lift_can")
    p.add_argument("--tasks", default="", help="Comma-separated task subset for multi_task; empty means all available")
    p.add_argument("--episodes", type=int, default=100, help="Episodes per task; 0 means all")
    p.add_argument("--action-mode", type=joint_action_mode, default="absolute_joint",
                   help="absolute[_joint] or relative[_joint]; relative arm targets use the current observation; gripper stays absolute")
    p.add_argument("--tactile-mode", choices=["none", "as_image", "encode", "native"], default="none")
    p.add_argument("--tactile-input-mode", choices=list(INPUT_CHANNELS), default="marker_rgb")
    p.add_argument("--tactile-type", choices=["cls", "full", "pool", "pool3x3", "pool5x5", "pool7x7"], default="cls",
                   help="cls: global token; full: all tokens; pool3x3/pool5x5/pool7x7: adaptive spatial pooling; pool aliases pool3x3")
    p.add_argument("--tactile-insert-location", choices=["encoder", "decoder"], default="decoder")
    p.add_argument('--tactile-preprocessing', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='Batch dense nonlinear TPS on CUDA; cpu retains reference preprocessing')
    p.add_argument('--wan-training-optimization', choices=['reference', 'compiled'], default='compiled',
                   help='Compile a full five-frame Wan encoder; keep all encoder parameters trainable')
    p.add_argument("--encoder-ckpt", type=Path)
    p.add_argument("--encoder-method", default="ResNet")
    p.add_argument('--encoder-pretrained-mode', choices=['univtac', 'tactile', 'vision'], default='univtac')
    p.add_argument('--encoder-pretrain-init', choices=['scratch', 'official', 'adapted'],
                   help='Require the encoder checkpoint to have this pretraining initialization')
    p.add_argument("--encoder-size", choices=["S", "B", "official"], default=None)
    p.add_argument("--cameras", choices=["auto", "head", "all"], default="auto")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--n-action-steps", type=int, default=30,
                   help="Actions executed before replanning (must be in [1, chunk-size])")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True, help="Global batch size, divided across DDP ranks")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument('--profile-stages', action='store_true', help='Synchronize each step and save detailed stage timings')
    p.add_argument('--profile-warmup', type=int, default=5)
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=5000)
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    p.add_argument("--max-val-batches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.chunk_size < 1:
        p.error("steps, batch-size, and chunk-size must be positive")
    if not 1 <= args.n_action_steps <= args.chunk_size:
        p.error("n-action-steps must be between 1 and chunk-size")
    if args.workers < 0 or args.log_freq < 1 or args.save_freq < 1:
        p.error("workers must be nonnegative; log-freq and save-freq must be positive")
    if args.profile_stages and not 0 <= args.profile_warmup < args.steps:
        p.error('profile-warmup must be nonnegative and smaller than steps')
    try:
        setup_distributed(args)
        with training_log(args):
            run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
