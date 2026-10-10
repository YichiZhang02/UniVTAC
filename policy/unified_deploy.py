"""Deployment bridge for checkpoints produced by train_policy.sh.

This module has no Isaac Sim imports. The evaluation harness imports Policy after
AppLauncher starts; offline inference can use the same observation adapter.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from ._base_policy import BasePolicy
from .checkpoints import ROOT, load_run
from .inference_assets import asset_path

sys.path.insert(0, str(ROOT / "encoder"))
from modality import INPUT_CHANNELS, tactile_from_arrays


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def training_image(value):
    """Match images read from existing UniVTAC HDF5 demonstrations.

    Collection passes sensor RGB arrays directly to OpenCV's BGR JPEG encoder.
    Training decodes those bytes and applies BGR2RGB, reversing sensor channels.
    Existing checkpoints therefore require the same reversal on live inputs.
    """
    image = as_numpy(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def rgb_tensor(value, image_size):
    image = training_image(value)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255


def make_vla_config(run, device, models_root):
    from policy.vla_common.engine.configs import FeatureType, PolicyFeature
    saved, options = run["config"], run["options"]
    features = {key: PolicyFeature(FeatureType.STATE if key == "observation.state" else FeatureType.VISUAL,
                                   tuple(shape)) for key, shape in saved["input_features"].items()}
    output_features = {key: PolicyFeature(FeatureType.ACTION, tuple(shape))
                       for key, shape in saved["output_features"].items()}
    cameras = run["camera_keys"]
    tactile_keys = [key for key in features if key.split(".")[-1].startswith("tac_")]
    common = dict(saved["base"], input_features=features, output_features=output_features,
        device=str(device), push_to_hub=False,
        top_camera_keys=[key for key in cameras if key.endswith("cam_high")],
        wrist_camera_keys=[key for key in cameras if key.endswith("cam_wrist")],
        tactile_keys=tactile_keys, tactile_mode=options["tactile_mode"],
        tactile_input_mode=options["tactile_input_mode"], tactile_type=options["tactile_type"],
        tactile_insert_location=options["tactile_insert_location"],
        tactile_encoder_path=None, tactile_encoder_config=saved.get("tactile_encoder_config"),
        state_mode=saved["state_mode"], action_mode=saved["action_mode"], action_gap=saved["action_gap"])
    common["chunk_size"] = saved.get("chunk_size", saved["base"].get("chunk_size", options["chunk_size"]))
    if run["policy"] == "starvla_groot":
        from policy.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
        bundled = asset_path(run, "qwen_path")
        common.update(base_vlm=str(bundled or models_root / "Qwen3.5-2B"),
                      vlm_from_config=bundled is not None, action_dim=8, state_dim=8)
        return StarvlaGrootConfig(**common)
    from policy.pi05.configuration_pi05 import PI05Config
    common["paligemma_tokenizer_path"] = str(asset_path(run, "tokenizer_path") or
        models_root / "pi05_base" / "paligemma-3b-pt-224-tokenizer")
    return PI05Config(**common)


class Policy(BasePolicy):
    def __init__(self, args):
        self.run = load_run(args["checkpoint"])
        self.task_name = args["task_name"]
        if self.task_name not in self.run["tasks"]:
            raise ValueError(f"Task {self.task_name} was not trained; available: {self.run['tasks']}")
        self.device = torch.device(args.get("device") or "cuda")
        self.options = self.run["options"]
        self.stats = self.run["stats"]
        self.kind = self.run["policy"]
        self.image_size = 256 if self.kind == "act" else self.options.get("image_size", 224)
        self.camera_keys = self.run["camera_keys"]
        self.tactile_mode = self.options.get("tactile_mode", "none")
        self.tactile_input_mode = self.options.get("tactile_input_mode", "marker_rgb")
        self.tokenizer = None
        models_root = Path(args.get("models_root") or ROOT / "resources" / "pretrained_models").expanduser().resolve()
        if self.kind == "act":
            from .ACT.act_policy import ACTPolicy
            saved = self.run["config"]
            config = {key: value for key, value in saved.items()
                      if key not in ("routing", "distributed", "tasks", "training_mode")}
            config.update(device=str(self.device), ckpt_dir=str(self.run["directory"]),
                          task_name=self.task_name, seed=0, num_epochs=1, tactile_ckpt=None)
            if self.tactile_mode == "encode":
                config["tactile_encoder_config"] = saved.get("tactile_encoder_config") or {
                    "method": config["tactile_encoder_method"], "size": config["tactile_encoder_size"],
                    "architecture": "resnet" if config["tactile_encoder_method"] == "ResNet" else "vit",
                    "input_mode": self.tactile_input_mode,
                    "in_chans": INPUT_CHANNELS[self.tactile_input_mode], "image_size": 224,
                }
            self.model = ACTPolicy(config).to(self.device)
            self.chunk_size = config["chunk_size"]
            self.temporal_agg = bool(config.get("temporal_agg", False))
            default_horizon = 1
        else:
            config = make_vla_config(self.run, self.device, models_root)
            if self.kind == "starvla_groot":
                from policy.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy
                self.model = StarvlaGrootPolicy(config).to(self.device)
            else:
                from transformers import AutoTokenizer
                from policy.pi05.modeling_pi05 import PI05Policy
                self.tokenizer = AutoTokenizer.from_pretrained(config.paligemma_tokenizer_path, local_files_only=True)
                self.model = PI05Policy(config).to(self.device)
            self.chunk_size = config.chunk_size
            self.temporal_agg = False
            default_horizon = config.n_action_steps

        # mmap keeps the optimizer tensors in the training bundle out of RAM.
        payload = torch.load(self.run["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        if "model" not in payload:
            raise ValueError("Expected a train_policy.sh checkpoint containing a 'model' state dict")
        if self.run["step"] is not None and payload.get("step") != self.run["step"]:
            raise ValueError("Checkpoint filename step does not match its saved training step")
        if payload.get("policy", self.kind) != self.kind:
            raise ValueError("Checkpoint policy does not match train_config.yml")
        state = payload["model"]
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        del state, payload
        self.execution_horizon = args.get("execution_horizon") or default_horizon
        if not 1 <= self.execution_horizon <= self.chunk_size:
            raise ValueError("Execution horizon must be between 1 and chunk_size")
        self.reset()
        print(f"Loaded {self.kind}: {self.run['checkpoint']} (step={self.run['step']})", flush=True)

    def normalize_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        stats = self.stats["state"]
        if self.kind == "pi05":
            low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
            return 2 * (state - low) / np.maximum(high - low, 1e-6) - 1
        return (state - np.asarray(stats["mean"], np.float32)) / np.asarray(stats["std"], np.float32)

    def encode_obs(self, observation):
        joint = as_numpy(observation["embodiment"]["joint"]).reshape(-1)[:8]
        if len(joint) != 8:
            raise ValueError("Expected at least eight joint values")
        batch = {"observation.state": torch.from_numpy(self.normalize_state(joint)).unsqueeze(0),
                 "task": [self.task_name.replace("_", " ")]}
        for key in self.camera_keys:
            camera = {"cam_high": "head", "cam_wrist": "wrist"}[key.split(".")[-1]]
            batch[key] = rgb_tensor(observation["observation"][camera]["rgb"], self.image_size).unsqueeze(0)
        if self.tactile_mode != "none":
            for side in ("left", "right"):
                stream = observation["tactile"][f"{side}_tactile"]
                if self.tactile_input_mode in ("rgb_only", "marker_rgb"):
                    key = "rgb" if self.tactile_input_mode == "rgb_only" else "rgb_marker"
                    raw = {key: training_image(stream[key])}
                else:
                    raw = {"marker": as_numpy(stream["marker"])}
                    if self.tactile_input_mode == "depth_deform":
                        raw["depth"] = as_numpy(stream["depth"])
                tactile = tactile_from_arrays(raw, self.tactile_input_mode, self.image_size)
                batch[f"observation.images.tac_{side}"] = tactile.unsqueeze(0)
        if self.tokenizer is not None:
            from .train_unified import make_pi_tokens
            make_pi_tokens(batch, self.tokenizer, self.model.config.tokenizer_max_length)
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()}

    @torch.no_grad()
    def predict_chunk(self, observation):
        batch = self.encode_obs(observation)
        if self.kind == "act":
            cameras = torch.stack([batch[key] for key in self.camera_keys], dim=1)
            mean = cameras.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
            std = cameras.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
            cameras = (cameras - mean) / std
            tactile = (torch.stack([batch[f"observation.images.tac_{side}"] for side in ("left", "right")], dim=1)
                       if self.tactile_mode != "none" else torch.empty(0, device=self.device))
            normalized = self.model(batch["observation.state"], cameras, tactile)
        else:
            normalized = self.model.predict_action_chunk(batch)
        if normalized.shape != (1, self.chunk_size, 8) or not torch.isfinite(normalized).all():
            raise ValueError(f"Invalid action chunk: shape={tuple(normalized.shape)}")
        stats = self.stats["action"]
        if self.kind == "pi05":
            low, high = normalized.new_tensor(stats["q01"]), normalized.new_tensor(stats["q99"])
            actions = (normalized + 1) * (high - low).clamp_min(1e-6) / 2 + low
        else:
            actions = normalized * normalized.new_tensor(stats["std"]) + normalized.new_tensor(stats["mean"])
        return actions[0].float()

    def eval(self, task, observation):
        if self.kind == "act" and self.temporal_agg:
            self.history.append((self.time, self.predict_chunk(observation)))
            self.history = [(start, chunk) for start, chunk in self.history if self.time - start < len(chunk)]
            candidates = torch.stack([chunk[self.time - start] for start, chunk in self.history])
            weights = torch.exp(-0.01 * torch.arange(len(candidates), device=candidates.device))
            action = (candidates * (weights / weights.sum())[:, None]).sum(0)
        else:
            if not self.actions:
                self.actions.extend(self.predict_chunk(observation)[:self.execution_horizon])
            action = self.actions.popleft()
        self.time += 1
        return task.take_action(action.to(task.device), action_type="qpos")

    def reset(self):
        self.actions = deque()
        self.history = []
        self.time = 0
        if hasattr(self.model, "reset"):
            self.model.reset()
