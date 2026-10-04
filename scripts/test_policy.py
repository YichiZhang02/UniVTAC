"""Unified testing CLI for train_policy.sh outputs and legacy deploy configs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.checkpoints import load_run


def hdf5_observation(path, frame, run):
    import cv2
    import h5py
    import numpy as np
    import torch
    observation = {"embodiment": {}, "observation": {}, "tactile": {}}
    with h5py.File(path, "r") as h5:
        if not 0 <= frame < len(h5["embodiment/joint"]):
            raise ValueError(f"Frame {frame} is outside {path}")
        observation["embodiment"]["joint"] = torch.from_numpy(h5["embodiment/joint"][frame].copy())

        def rgb(key):
            image = cv2.imdecode(np.frombuffer(h5[key][frame], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {path}:{key}[{frame}]")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for key in run["camera_keys"]:
            camera = {"cam_high": "head", "cam_wrist": "wrist"}[key.split(".")[-1]]
            observation["observation"][camera] = {"rgb": rgb(f"observation/{camera}/rgb")}
        if run["options"].get("tactile_mode", "none") != "none":
            mode = run["options"].get("tactile_input_mode", "marker_rgb")
            for side in ("left", "right"):
                prefix = f"tactile/{side}_tactile"
                if mode in ("rgb_only", "marker_rgb"):
                    key = "rgb" if mode == "rgb_only" else "rgb_marker"
                    data = {key: rgb(f"{prefix}/{key}")}
                else:
                    data = {"marker": h5[f"{prefix}/marker"][frame]}
                    if mode == "depth_deform":
                        data["depth"] = h5[f"{prefix}/depth"][frame]
                observation["tactile"][f"{side}_tactile"] = data
    return observation


def inspect_run(run):
    return {"policy": run["policy"], "experiment": run["label"], "step": run["step"],
            "checkpoint": str(run["checkpoint"]), "size_GiB": round(run["checkpoint"].stat().st_size / 1024**3, 2),
            "tasks": run["tasks"], "camera_keys": run["camera_keys"],
            "tactile_mode": run["options"].get("tactile_mode", "none"),
            "tactile_input_mode": run["options"].get("tactile_input_mode"),
            "encoder_method": run["options"].get("encoder_method"),
            "encoder_size": run["options"].get("encoder_size"),
            "tactile_type": run["options"].get("tactile_type"),
            "tactile_insert_location": run["options"].get("tactile_insert_location"),
            "config": str(run["directory"] / "train_config.yml"),
            "stats": str(run["directory"] / "dataset_stats.json"),
            "note": "Metadata/file checks only; --offline loads weights and predicts one HDF5 frame."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Examples: bash test_policy.sh --checkpoint RUN_DIR --inspect; "
        "bash test_policy.sh --checkpoint RUN_DIR --task lift_can --offline; "
        "bash test_policy.sh --checkpoint RUN_DIR --task lift_can --headless; "
        "legacy: bash test_policy.sh TASK TASK_CONFIG DEPLOY_CONFIG [GPU]. "
        "The 5090 simulator testing environment remains unverified."))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT") or os.environ.get("CKPT_DIR"),
                        help="train_policy.sh run directory or checkpoint_<step>.pt; directory selects latest step")
    parser.add_argument("--step", type=int, help="Choose a checkpoint step within a run directory")
    parser.add_argument("--task", default=os.environ.get("TASK"), help="Task name; defaults to the first trained task")
    parser.add_argument("--task-config", default=os.environ.get("TASK_CONFIG", "demo"))
    parser.add_argument("--gpu", default=os.environ.get("GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--models-root", type=Path, default=ROOT / "resources" / "pretrained_models")
    parser.add_argument("--execution-horizon", type=int, help="Number of actions executed before replanning")
    parser.add_argument("--total-num", "--total_num", type=int, default=100)
    parser.add_argument("--inspect", action="store_true", help="Inspect run metadata without loading weights or Isaac Sim")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved deployment configuration without starting simulation")
    parser.add_argument("--offline", action="store_true", help="Load checkpoint and predict one recorded HDF5 frame without simulation")
    parser.add_argument("--hdf5", type=Path, help="Explicit offline HDF5 episode")
    parser.add_argument("--data-root", type=Path, default=ROOT / "resources" / "data" / "isaac51")
    parser.add_argument("--frame", type=int, default=0)
    argv = sys.argv[1:]
    legacy = []
    while argv and not argv[0].startswith("-"):
        legacy.append(argv.pop(0))
    args, simulator_args = parser.parse_known_args(argv)
    args.legacy = legacy
    if sum((args.inspect, args.dry_run, args.offline)) > 1:
        parser.error("Choose one of --inspect, --dry-run, or --offline")
    if args.total_num < 1 or (args.execution_horizon is not None and args.execution_horizon < 1):
        parser.error("total-num and execution-horizon must be positive")
    if args.legacy:
        if len(args.legacy) not in (3, 4) or args.checkpoint:
            parser.error("Legacy form is TASK TASK_CONFIG DEPLOY_CONFIG [GPU], without --checkpoint")
        if args.inspect or args.offline:
            parser.error("--inspect/--offline require a unified --checkpoint")
        task, task_config, deploy_file = args.legacy[:3]
        if len(args.legacy) == 4:
            args.gpu = args.legacy[3]
        deploy = None
    else:
        if not args.checkpoint:
            parser.error("Provide --checkpoint RUN_DIR/FILE (or CHECKPOINT=...), or legacy deploy arguments")
        run = load_run(args.checkpoint, args.step)
        task = args.task or run["tasks"][0]
        if task not in run["tasks"]:
            parser.error(f"Task {task} was not trained; available: {run['tasks']}")
        if args.inspect:
            print(json.dumps(inspect_run(run), indent=2, ensure_ascii=False))
            return 0
        task_config = args.task_config
        deploy = {"policy_name": "unified_deploy", "checkpoint": str(run["checkpoint"]),
                  "device": args.device, "models_root": str(args.models_root.expanduser().resolve()),
                  "seed": 0, "instruction_type": "seen"}
        if args.execution_horizon:
            deploy["execution_horizon"] = args.execution_horizon

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.offline:
        if simulator_args:
            parser.error(f"Unknown offline arguments: {simulator_args}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        hdf5_path = args.hdf5
        if hdf5_path is None:
            episodes = json.loads((run["directory"] / "episodes.json").read_text())
            files = episodes.get("val", {}).get(task) or episodes.get("train", {}).get(task)
            if not files:
                raise FileNotFoundError(f"No recorded episode filenames for task {task}")
            hdf5_path = args.data_root / task / "hdf5" / files[0]
        from policy.unified_deploy import Policy
        policy = Policy({**deploy, "task_name": task})
        actions = policy.predict_chunk(hdf5_observation(hdf5_path, args.frame, run))
        print(json.dumps({"mode": "offline", "task": task, "episode": str(hdf5_path),
                          "frame": args.frame, "checkpoint_step": run["step"],
                          "action_shape": list(actions.shape), "first_action": actions[0].cpu().tolist(),
                          "note": "Checkpoint inference check; this is not a simulator success-rate evaluation."}, indent=2))
        return 0

    if args.dry_run:
        print(json.dumps({"task": task, "task_config": task_config, "gpu": args.gpu,
                          "total_num": args.total_num, "deploy": deploy if deploy else deploy_file,
                          "simulator_args": simulator_args}, indent=2))
        return 0
    with tempfile.TemporaryDirectory(prefix="univtac-test-") as folder:
        if deploy is not None:
            import yaml
            deploy_file = str(Path(folder) / "deploy.yml")
            Path(deploy_file).write_text(yaml.safe_dump(deploy, sort_keys=False))
        command = [sys.executable, str(ROOT / "scripts" / "eval_policy.py"),
                   task, task_config, deploy_file, "--device", args.device,
                   "--total_num", str(args.total_num), *simulator_args]
        print("Launching simulator test: " + shlex.join(command), flush=True)
        return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
