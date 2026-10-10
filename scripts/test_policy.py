"""Unified testing CLI for train_policy.sh outputs and legacy deploy configs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.checkpoints import load_run


def hdf5_observation(path, frame, run):
    """Reconstruct sensor-format observations for the shared deployment adapter."""
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
            # Collection encoded sensor arrays directly with OpenCV. Decoding
            # without conversion restores their original channel order; the
            # deployment adapter then applies the training channel convention.
            return image

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


def run_batch(args, simulator_args, parser):
    if not args.model_id or Path(args.model_id).name != args.model_id or args.model_id in (".", ".."):
        parser.error("--model-id must be a single directory name under policy_results")
    if args.step is None or args.step < 0:
        parser.error("Batch evaluation requires --step with a nonnegative checkpoint step")
    if args.num_gpu is None or args.num_gpu < 1:
        parser.error("Batch evaluation requires --num-gpu >= 1")
    visible = args.cuda_visible_devices
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or args.num_gpu > len(gpu_ids):
        parser.error("Set --cuda-visible-devices or CUDA_VISIBLE_DEVICES to at least --num-gpu distinct GPU IDs")
    gpu_ids = gpu_ids[:args.num_gpu]

    run = load_run(ROOT / "policy_results" / args.model_id, args.step)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()] if args.tasks else list(run["tasks"])
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in run["tasks"] for task in tasks):
        parser.error(f"--tasks must contain unique tasks from {run['tasks']}")
    result_dir = ROOT / "test_results" / args.model_id / str(args.step)
    summary = {"model_id": args.model_id, "step": args.step, "checkpoint": str(run["checkpoint"]),
               "task_config": args.task_config, "tasks": tasks, "gpu_ids": gpu_ids,
               "total_num_per_task": args.total_num, "start_seed": args.start_seed,
               "resolved_start_seed": 1000000 if args.start_seed == -1 else args.start_seed,
               "max_seed": args.max_seed, "max_errors_per_task": args.max_errors,
               "instruction_type": "seen", "headless": True, "simulator_args": simulator_args,
               "result_dir": str(result_dir)}
    if args.dry_run or args.inspect:
        print(json.dumps(summary, indent=2))
        return 0
    if result_dir.exists():
        if not result_dir.is_dir():
            raise ValueError(f"Test output path is not a directory: {result_dir}")
        if any(result_dir.iterdir()):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            archive = result_dir.with_name(f"{result_dir.name}_previous_{timestamp}")
            suffix = 1
            while archive.exists():
                archive = result_dir.with_name(f"{result_dir.name}_previous_{timestamp}_{suffix}")
                suffix += 1
            result_dir.rename(archive)
            summary["archived_previous_run"] = str(archive)
            print(f"Archived previous test outputs: {archive}", flush=True)

    for subdir in ("logs", "videos", "results"):
        (result_dir / subdir).mkdir(parents=True, exist_ok=True)
    deploy_file = result_dir / "deploy.yml"
    import yaml
    deploy = {"policy_name": "unified_deploy", "checkpoint": str(run["checkpoint"]),
              "device": args.device, "models_root": str(args.models_root.expanduser().resolve()),
              "seed": 0, "instruction_type": "seen"}
    if args.execution_horizon:
        deploy["execution_horizon"] = args.execution_horizon
    deploy_file.write_text(yaml.safe_dump(deploy, sort_keys=False))
    (result_dir / "settings.json").write_text(json.dumps(summary, indent=2) + "\n")

    def worker(task, gpu_id):
        log_file = result_dir / "logs" / f"{task}.log"
        result_file = result_dir / "results" / f"{task}.json"
        video_dir = result_dir / "videos" / task
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.pop("LD_LIBRARY_PATH", None)
        glu_library = Path(sys.prefix) / "lib" / "libGLU.so.1"
        if glu_library.is_file():
            env.setdefault("LD_PRELOAD", str(glu_library))
        command = [sys.executable, "-u", str(ROOT / "scripts" / "eval_policy.py"),
                   task, args.task_config, str(deploy_file), "--device", args.device,
                   "--total_num", str(args.total_num), "--start_seed", str(args.start_seed),
                   "--max_seed", str(args.max_seed), "--max-errors", str(args.max_errors),
                   "--headless", "--save-dir", str(video_dir),
                   "--result-json", str(result_file), *simulator_args]
        started = time.time()
        with log_file.open("w") as stream:
            stream.write("Command: " + shlex.join(command) + "\nGPU: " + gpu_id + "\n")
            stream.flush()
            try:
                status = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT).returncode
            except OSError as error:
                stream.write(f"Launch failed: {error}\n")
                status = -1
        try:
            counts = json.loads(result_file.read_text())
        except (OSError, ValueError):
            counts = {}
        valid = int(counts.get("test_num", 0))
        success = int(counts.get("succ_num", 0))
        complete = status == 0 and valid == args.total_num
        return {"task": task, "gpu": gpu_id, "exit_code": status,
                "success": success, "evaluated": valid, "errors": int(counts.get("error_num", 0)),
                "requested": args.total_num, "success_rate": success / valid if valid else None,
                "complete": complete, "seconds": round(time.time() - started, 1),
                "log": str(log_file.relative_to(result_dir)),
                "videos": str(video_dir.relative_to(result_dir))}

    pending = iter(tasks)
    results = {}
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        active = {}
        for gpu_id in gpu_ids:
            task = next(pending, None)
            if task is not None:
                print(f"Starting {task} on GPU {gpu_id}", flush=True)
                active[pool.submit(worker, task, gpu_id)] = gpu_id
        while active:
            future = next(as_completed(active))
            gpu_id = active.pop(future)
            result = future.result()
            results[result["task"]] = result
            print(f"Finished {result['task']} on GPU {gpu_id}: "
                  f"{result['success']}/{result['evaluated']} "
                  f"({'complete' if result['complete'] else 'incomplete'})", flush=True)
            task = next(pending, None)
            if task is not None:
                print(f"Starting {task} on GPU {gpu_id}", flush=True)
                active[pool.submit(worker, task, gpu_id)] = gpu_id

    ordered = [results[task] for task in tasks]
    all_complete = all(item["complete"] for item in ordered)
    average = sum(item["success_rate"] for item in ordered) / len(ordered) if all_complete else None
    report = {**summary, "results": ordered, "average_success_rate": average,
              "all_complete": all_complete}
    (result_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Task | GPU | Success | Rate | Status |", "| --- | --- | ---: | ---: | --- |"]
    for item in ordered:
        rate = f"{item['success_rate'] * 100:.2f}%" if item["success_rate"] is not None else "N/A"
        lines.append(f"| {item['task']} | {item['gpu']} | {item['success']}/{item['evaluated']} "
                     f"| {rate} | {'complete' if item['complete'] else 'incomplete'} |")
    lines.append(f"\nAverage success rate: {average * 100:.2f}%" if all_complete
                 else "\nAverage success rate: N/A (some tasks incomplete)")
    text = "\n".join(lines) + "\n"
    (result_dir / "summary.md").write_text(text)
    print(text + f"Results: {result_dir}")
    return 0 if all_complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Batch: bash test_policy.sh --model-id RUN_NAME --step 30000 "
        "--cuda-visible-devices 0,2 --num-gpu 2; "
        "Examples: bash test_policy.sh --checkpoint RUN_DIR --inspect; "
        "bash test_policy.sh --checkpoint RUN_DIR --task lift_can --offline; "
        "bash test_policy.sh --checkpoint RUN_DIR --task lift_can --headless; "
        "legacy: bash test_policy.sh TASK TASK_CONFIG DEPLOY_CONFIG [GPU]. "
        "The 5090 simulator testing environment remains unverified."))
    parser.add_argument("--checkpoint",
                        help="train_policy.sh run directory or checkpoint_<step>.pt; directory selects latest step")
    parser.add_argument("--model-id", help="Run directory name under policy_results; evaluates all tasks")
    parser.add_argument("--step", type=int, help="Choose a checkpoint step within a run directory")
    parser.add_argument("--cuda-visible-devices",
                        help="Comma-separated GPU IDs for batch evaluation; overrides CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num-gpu", type=int, help="Number of CUDA_VISIBLE_DEVICES GPUs to schedule")
    parser.add_argument("--tasks", help="Optional comma-separated subset for batch evaluation")
    parser.add_argument("--task", default=os.environ.get("TASK"), help="Task name; defaults to the first trained task")
    parser.add_argument("--task-config", default=os.environ.get("TASK_CONFIG", "demo"))
    parser.add_argument("--gpu", default=os.environ.get("GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--models-root", type=Path, default=ROOT / "resources" / "pretrained_models")
    parser.add_argument("--execution-horizon", type=int, help="Number of actions executed before replanning")
    parser.add_argument("--total-num", "--total_num", type=int, default=100)
    parser.add_argument("--start-seed", "--start_seed", type=int, default=-1)
    parser.add_argument("--max-seed", "--max_seed", type=int, default=-1)
    parser.add_argument("--max-errors", type=int, default=10,
                        help="Batch task stops after this many evaluation errors (default: 10)")
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
    if args.total_num < 1 or args.max_errors < 1 or (args.execution_horizon is not None and args.execution_horizon < 1):
        parser.error("total-num and execution-horizon must be positive")
    if not args.model_id:
        args.checkpoint = args.checkpoint or os.environ.get("CHECKPOINT") or os.environ.get("CKPT_DIR")
    if args.model_id:
        if args.checkpoint or args.legacy or args.offline:
            parser.error("--model-id cannot be combined with --checkpoint, legacy arguments, or --offline")
        return run_batch(args, simulator_args, parser)
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
                   "--total_num", str(args.total_num), "--start_seed", str(args.start_seed),
                   "--max_seed", str(args.max_seed), *simulator_args]
        print("Launching simulator test: " + shlex.join(command), flush=True)
        return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
