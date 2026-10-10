"""Train ACT, PI0.5, or StarVLA-GR00T on UniVTAC raw HDF5 demonstrations."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
import sys
import traceback
from collections import defaultdict
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
from modality import INPUT_CHANNELS, read_tactile  # noqa: E402


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


def compute_stats(files_by_task: dict[str, list[Path]]) -> dict:
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
    return result


class UniVTACDataset(Dataset):
    def __init__(self, files_by_task, *, cameras, tactile_mode, tactile_input_mode,
                 chunk_size, stats, policy, image_size):
        self.rows = []
        self.task_counts = defaultdict(int)
        self.cameras = cameras
        self.tactile_mode = tactile_mode
        self.tactile_input_mode = tactile_input_mode
        self.chunk_size = chunk_size
        self.stats = stats
        self.policy = policy
        self.image_size = image_size
        self._files = {}
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
        state["_files"] = {}
        return state

    def _file(self, path):
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def _norm(self, x, name):
        s = self.stats[name]
        if self.policy == "pi05":
            low = np.asarray(s["q01"], dtype=np.float32)
            high = np.asarray(s["q99"], dtype=np.float32)
            return 2 * (x - low) / np.maximum(high - low, 1e-6) - 1
        return (x - np.asarray(s["mean"], dtype=np.float32)) / np.asarray(s["std"], dtype=np.float32)

    def _rgb(self, f, key, t):
        encoded = np.frombuffer(f[key][t], dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode {f.filename}:{key}[{t}]")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                tactile = read_tactile(f, side, t, self.tactile_input_mode, self.image_size)
                if self.tactile_mode == "as_image" and tactile.shape[0] == 1:
                    tactile = tactile.repeat(3, 1, 1)
                out[f"observation.images.tac_{side}"] = tactile
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
                  tactile_encoder_path=str(args.encoder_ckpt) if args.tactile_mode == "encode" else None,
                  tactile_insert_location=args.tactile_insert_location,
                  state_mode="absolute_joint", action_mode="absolute_joint", action_gap=1,
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
    return StarvlaGrootConfig(**common)


def make_pi_tokens(batch, tokenizer, max_length):
    state = batch["observation.state"].detach().cpu().numpy()
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(state, bins=bins) - 1
    prompts = [f"Task: {task}, State: {' '.join(map(str, row))};\nAction: "
               for task, row in zip(batch["task"], discretized)]
    encoded = tokenizer(prompts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    batch["observation.language.tokens"] = encoded["input_ids"]
    batch["observation.language.attention_mask"] = encoded["attention_mask"].bool()


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.tactile_mode == "encode" and not args.encoder_ckpt:
        raise ValueError("encode mode requires --encoder-ckpt")
    tasks = select_tasks(args)
    train_files, val_files = split_files(tasks, args.episodes, args.seed)
    if args.world_size > 1:
        stats_holder = [compute_stats(train_files) if args.rank == 0 else None]
        dist.broadcast_object_list(stats_holder, src=0)
        stats = stats_holder[0]
    else:
        stats = compute_stats(train_files)
    settings = json.loads((ROOT / "policy" / "task_settings.json").read_text())
    if args.cameras == "auto":
        cameras = ["head", "wrist"] if all(settings[t].get("camera_type") == "all" for t in tasks) else ["head"]
    else:
        cameras = ["head", "wrist"] if args.cameras == "all" else ["head"]
    image_size = 256 if args.policy == "act" else args.image_size
    make_ds = lambda files: UniVTACDataset(files, cameras=cameras, tactile_mode=args.tactile_mode,
                                           tactile_input_mode=args.tactile_input_mode,
                                           chunk_size=args.chunk_size, stats=stats,
                                           policy=args.policy, image_size=image_size)
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
    loader = DataLoader(train_ds, batch_size=args.batch_size_per_rank, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed + args.rank), **loader_options)
    if len(loader) == 0:
        raise ValueError("Training dataset is smaller than one batch")
    # Validation is run by rank 0 on the unwrapped model, at the per-GPU batch size.
    # Load validation synchronously: capped validation exits early, and worker
    # teardown with prefetched batches can abort during iterator cleanup.
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size_per_rank, num_workers=0)
                  if len(val_ds) and args.rank == 0 else None)
    output = args.output_dir
    camera_keys = [f"observation.images.cam_{'high' if c == 'head' else 'wrist'}" for c in cameras]
    tactile_keys = [f"observation.images.tac_{side}" for side in ("left", "right")]
    if args.policy == "act":
        sys.path.insert(0, str(ROOT / "policy" / "ACT"))
        from act_policy import ACTPolicy
        base = yaml.safe_load((ROOT / "policy" / "ACT" / "base_config.yml").read_text())
        base.update(chunk_size=args.chunk_size, camera_names=["cam_high" if c == "head" else "cam_wrist" for c in cameras],
                    tactile_names=["tac_left", "tac_right"] if args.tactile_mode != "none" else [],
                    tactile_ckpt=str(args.encoder_ckpt) if args.tactile_mode == "encode" else None,
                    tactile_input_mode=args.tactile_input_mode, tactile_type=args.tactile_type,
                    batch_size=args.batch_size, num_steps=args.steps, device=args.device,
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
            config_dump["tactile_encoder_config"] = {key: encoder_state[key] for key in
                ("method", "size", "architecture", "image_size", "input_mode", "in_chans") if key in encoder_state}
        tokenizer = None
    else:
        base = yaml.safe_load((ROOT / "policy" / args.policy / "base_config.yml").read_text())
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
            model = StarvlaGrootPolicy(config)
        model.to(args.device)
        # Keep update/clipping order stable when a policy orders its modules
        # differently to overlap DDP communication with backward computation.
        optim_params = list(model.get_optim_params())
        optimizer = torch.optim.AdamW(optim_params, lr=config.optimizer_lr,
                                      betas=tuple(config.optimizer_betas), eps=config.optimizer_eps,
                                      weight_decay=config.optimizer_weight_decay)
        scheduler = config.get_scheduler_preset().build(optimizer, args.steps)
        config_dump = {"policy": args.policy, "base": base,
                       "chunk_size": config.chunk_size,
                       "routing": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       "state_mode": "absolute_joint", "action_mode": "absolute_joint", "action_gap": 1,
                       "tactile_encoder_config": config.tactile_encoder_config,
                       "input_features": {k: list(v.shape) for k, v in config.input_features.items()},
                       "output_features": {k: list(v.shape) for k, v in config.output_features.items()}}
    config_dump["distributed"] = {"world_size": args.world_size, "global_batch_size": args.batch_size,
                                   "batch_size_per_rank": args.batch_size_per_rank}
    if args.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        from policy.inference_assets import export_assets
        inference = export_assets(output, args.policy,
            qwen=model.qwen_vl if args.policy == "starvla_groot" else None,
            tokenizer=tokenizer)
        if inference is not None:
            config_dump["inference"] = inference
        (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
        (output / "episodes.json").write_text(json.dumps({"train": {k: [p.name for p in v] for k, v in train_files.items()},
                                                            "val": {k: [p.name for p in v] for k, v in val_files.items()}}, indent=2))
        (output / "train_config.yml").write_text(yaml.safe_dump(config_dump, sort_keys=False))
        print(f"Tasks: {tasks}; train rows={len(train_ds)}; val rows={len(val_ds)}; cameras={cameras}; "
              f"world_size={args.world_size}; global_batch={args.batch_size}; per_gpu_batch={args.batch_size_per_rank}", flush=True)
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

    def batch_loss(raw, forward_model):
        if tokenizer is not None:
            make_pi_tokens(raw, tokenizer, config.tokenizer_max_length)
        batch = {k: v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in raw.items()}
        if args.policy == "act":
            cam = torch.stack([batch[k] for k in camera_keys], dim=1)
            mean = torch.tensor([0.485, 0.456, 0.406], device=cam.device)[None, None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=cam.device)[None, None, :, None, None]
            cam = (cam - mean) / std
            tac = torch.stack([batch[k] for k in tactile_keys], dim=1) if args.tactile_mode != "none" else torch.empty(0, device=args.device)
            return forward_model(batch["observation.state"], cam, tac, batch["action"], batch["is_pad"])["loss"]
        return forward_model(batch)[0]

    epoch = 0
    train_iter = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_iter = iter(loader)
            batch = next(train_iter)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(batch, model)
        loss.backward()
        if args.policy != "act":
            torch.nn.utils.clip_grad_norm_(optim_params, config.optimizer_grad_clip_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        if step == 1 or step % args.log_freq == 0:
            mean_loss = loss.detach().float().clone()
            if args.world_size > 1:
                dist.all_reduce(mean_loss)
                mean_loss /= args.world_size
            if args.rank == 0:
                print(f"step={step} loss={float(mean_loss):.6f} lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
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
                torch.save({"format_version": 1, "policy": args.policy,
                            "model": core_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "step": step}, output / f"checkpoint_{step}.pt")
            if args.world_size > 1:
                dist.barrier()
    if args.rank == 0:
        print(f"Finished: {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["act", "pi05", "starvla_groot"], required=True)
    p.add_argument("--training-mode", choices=["single_task", "multi_task"], default="single_task")
    p.add_argument("--task", default="lift_can")
    p.add_argument("--tasks", default="", help="Comma-separated task subset for multi_task; empty means all available")
    p.add_argument("--episodes", type=int, default=100, help="Episodes per task; 0 means all")
    p.add_argument("--tactile-mode", choices=["none", "as_image", "encode"], default="none")
    p.add_argument("--tactile-input-mode", choices=list(INPUT_CHANNELS), default="marker_rgb")
    p.add_argument("--tactile-type", choices=["cls", "full"], default="cls")
    p.add_argument("--tactile-insert-location", choices=["encoder", "decoder"], default="decoder")
    p.add_argument("--encoder-ckpt", type=Path)
    p.add_argument("--encoder-method", default="ResNet")
    p.add_argument("--encoder-size", choices=["S", "B"], default="S")
    p.add_argument("--cameras", choices=["auto", "head", "all"], default="auto")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--chunk-size", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True, help="Global batch size, divided across DDP ranks")
    p.add_argument("--workers", type=int, default=4)
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
    if args.workers < 0 or args.log_freq < 1 or args.save_freq < 1:
        p.error("workers must be nonnegative; log-freq and save-freq must be positive")
    try:
        setup_distributed(args)
        with training_log(args):
            run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
