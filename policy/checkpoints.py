"""Resolve unified training checkpoints and their portable run metadata."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP_PATTERN = re.compile(r"checkpoint_(\d+)\.pt$")


def training_options(config):
    if "routing" in config:
        return dict(config["routing"])
    options = dict(config)
    options.setdefault("policy", "act")
    options.setdefault("training_mode", "single_task")
    options.setdefault("tactile_input_mode", config.get("tactile_input_mode", "marker_rgb"))
    options.setdefault("encoder_method", config.get("tactile_encoder_method", "ResNet"))
    options.setdefault("encoder_size", config.get("tactile_encoder_size", "S"))
    options.setdefault("tactile_insert_location", "decoder")
    return options


def run_label(config):
    options = training_options(config)
    policy = options.get("policy", config.get("policy", "act"))
    mode = options.get("training_mode", "single_task")
    if mode == "multi_task":
        tasks = options.get("tasks", "")
        if isinstance(tasks, list):
            tasks = ",".join(tasks)
        task_label = tasks.replace(",", "_") or "all"
    else:
        task_label = options.get("task") or config.get("tasks", ["unknown"])[0]
    tactile_mode = options.get("tactile_mode", "none")
    parts = [policy, mode, task_label, tactile_mode]
    if tactile_mode != "none":
        parts.append(options.get("tactile_input_mode", "marker_rgb"))
    if tactile_mode == "encode":
        parts.extend([options.get("encoder_method", "ResNet"), options.get("encoder_size", "S"),
                      options.get("tactile_type", "cls"), options.get("tactile_insert_location", "decoder")])
    label = "_".join(parts)
    if "/" in label or "\\" in label:
        raise ValueError(f"Invalid experiment label: {label}")
    return label


def resolve_checkpoint(path, step=None):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if step is not None:
            raise ValueError("--step requires a training directory, not a checkpoint file")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint file or training directory does not exist: {path}")
    if step is not None:
        checkpoint = path / f"checkpoint_{step}.pt"
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise FileNotFoundError(checkpoint)
        return checkpoint
    candidates = [(int(match.group(1)), child) for child in path.glob("checkpoint_*.pt")
                  if (match := STEP_PATTERN.fullmatch(child.name)) and child.stat().st_size]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_<step>.pt files in {path}")
    return max(candidates, key=lambda item: item[0])[1]


def load_run(path, step=None):
    import yaml
    checkpoint = resolve_checkpoint(path, step)
    folder = checkpoint.parent
    config_path, stats_path = folder / "train_config.yml", folder / "dataset_stats.json"
    if not config_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Keep train_config.yml and dataset_stats.json beside {checkpoint.name}")
    config = yaml.safe_load(config_path.read_text())
    stats = json.loads(stats_path.read_text())
    for name in ("state", "action"):
        for key in ("mean", "std", "q01", "q99"):
            if len(stats.get(name, {}).get(key, [])) != 8:
                raise ValueError(f"{stats_path}: expected eight values for {name}.{key}")
        if any(value <= 0 for value in stats[name]["std"]):
            raise ValueError(f"{stats_path}: {name}.std must be positive")
    options = training_options(config)
    policy = options.get("policy", config.get("policy", "act"))
    if policy not in ("act", "pi05", "starvla_groot"):
        raise ValueError(f"Unsupported unified checkpoint policy: {policy}")
    if "input_features" in config:
        camera_keys = [key for key in config["input_features"]
                       if key.startswith("observation.images.") and not key.split(".")[-1].startswith("tac_")]
    else:
        camera_keys = [f"observation.images.{camera}" for camera in config["camera_names"]]
    tasks = config.get("tasks")
    episodes_path = folder / "episodes.json"
    if episodes_path.is_file():
        episodes = json.loads(episodes_path.read_text())
        tasks = list(episodes.get("train", {}))
    elif not tasks:
        requested = options.get("tasks", "")
        if options.get("training_mode") == "multi_task":
            if isinstance(requested, str):
                tasks = [name.strip() for name in requested.split(",") if name.strip()]
            else:
                tasks = list(requested)
            if not tasks:
                # Older VLA exports omitted both the selected task list and
                # episodes.json. The simulator's instruction files enumerate
                # the eight supported tasks; task_settings also contains
                # training-only entries without simulator environments.
                instruction_dir = ROOT / "resources" / "instructions"
                tasks = sorted(path.stem for path in instruction_dir.glob("*.json")
                               if (ROOT / "resources" / "envs" / f"{path.stem}.py").is_file())
        else:
            tasks = [options.get("task", "lift_can")]
    match = STEP_PATTERN.fullmatch(checkpoint.name)
    return {"checkpoint": checkpoint, "directory": folder, "config": config, "stats": stats,
            "options": options, "policy": policy, "camera_keys": camera_keys, "tasks": tasks,
            "step": int(match.group(1)) if match else None, "label": run_label(config)}
