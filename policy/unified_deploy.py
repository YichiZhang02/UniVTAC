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
from encoder.image_io import require_rgb_model
from .joint_actions import decode_joint_actions

sys.path.insert(0, str(ROOT / "encoder"))
from modality import DEPTH_DEFORM_SPECS, INPUT_CHANNELS, tactile_from_arrays


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def rgb_image(value):
    """Live sensor images already use the model's RGB convention."""
    image = as_numpy(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {image.shape}")
    return image


def rgb_tensor(value, image_size):
    image = rgb_image(value)
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
        tactile_encoder_pretrained_mode=saved.get('encoder_pretrained_mode', 'univtac'),
        tactile_num_frames=saved.get('tactile_num_frames', 1),
        tactile_frame_offset=saved.get('tactile_frame_stride', 1),
        state_mode=saved["state_mode"], action_mode=options["action_mode"], action_gap=saved["action_gap"])
    common["chunk_size"] = saved.get("chunk_size", saved["base"].get("chunk_size", options["chunk_size"]))
    if "n_action_steps" in saved:
        common["n_action_steps"] = saved["n_action_steps"]
    if run["policy"] in ("starvla_groot", "starvla_pi"):
        from policy.starvla_groot.architecture import restore_starvla_architecture
        common = restore_starvla_architecture(common, run["policy"])
        from policy.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
        bundled = asset_path(run, "qwen_path")
        # Inference defaults supersede historical training/checkpoint settings.
        # Policy args below can still explicitly override them for deployment.
        common.update(base_vlm=str(bundled or models_root / "Qwen3.5-2B"),
                      vlm_from_config=bundled is not None, action_dim=8, state_dim=8,
                      use_cuda_graph=True, repeated_diffusion_steps=1)
        if run["policy"] == "starvla_pi":
            from policy.starvla_pi import StarvlaPiConfig
            return StarvlaPiConfig(**common)
        return StarvlaGrootConfig(**common)
    from policy.pi05.configuration_pi05 import PI05Config
    common["paligemma_tokenizer_path"] = str(asset_path(run, "tokenizer_path") or
        models_root / "pi05_base" / "paligemma-3b-pt-224-tokenizer")
    return PI05Config(**common)


class Policy(BasePolicy):
    def __init__(self, args):
        self.run = load_run(args["checkpoint"])
        require_rgb_model(self.run["config"])
        self.task_name = args["task_name"]
        if self.task_name not in self.run["tasks"]:
            raise ValueError(f"Task {self.task_name} was not trained; available: {self.run['tasks']}")
        self.device = torch.device(args.get("device") or "cuda")
        self.options = self.run["options"]
        self.action_mode = self.options["action_mode"]
        self.stats = self.run["stats"]
        self.kind = self.run["policy"]
        self.image_size = 256 if self.kind == "act" else self.options.get("image_size", 224)
        self.camera_keys = self.run["camera_keys"]
        self.tactile_mode = self.options.get("tactile_mode", "none")
        self.tactile_input_mode = self.options.get("tactile_input_mode", "marker_rgb")
        self.encoder_pretrained_mode = self.run['config'].get('encoder_pretrained_mode', 'univtac')
        self.tactile_num_frames = self.run['config'].get('tactile_num_frames', 1)
        self.tactile_frame_stride = self.run['config'].get('tactile_frame_stride', 1)
        self.tactile_simulation_step_stride = self.run['config'].get(
            'tactile_simulation_step_stride', (self.run['config'].get('tactile_encoder_config') or {}).get('simulation_step_stride'))
        if self.tactile_simulation_step_stride is not None and (
                self.tactile_num_frames != 4 or self.tactile_frame_stride != 1 or self.tactile_simulation_step_stride != 2):
            raise ValueError('Temporal encoder requires 4 frames at exactly 2 physical simulation steps')
        self.tactile_background_offset = (130/255 if self.options.get('encoder_method') == 'AnyTouch2' else .5)
        if self.tactile_mode != "none":
            from encoder.linear_depth_deform import require_depth_deform_encoding
            require_depth_deform_encoding(self.run["config"], self.tactile_input_mode)
        self.record_policy_video = args.get("record_policy_video", False)
        self.save_policy_inputs = args.get("save_policy_inputs", False)
        self.tokenizer = None
        models_root = Path(args.get("models_root") or ROOT / "resources" / "pretrained_models").expanduser().resolve()
        if self.kind == "act":
            from .ACT.act_policy import ACTPolicy
            saved = self.run["config"]
            config = {key: value for key, value in saved.items()
                      if key not in ("routing", "distributed", "tasks", "training_mode", "color_order")}
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
            # ACT defaults to one action step, regardless of checkpoint n_action_steps.
            default_horizon = 1
            if "n_action_steps" in args:
                horizon = args["n_action_steps"]
                if type(horizon) is not int or not 1 <= horizon <= self.chunk_size:
                    raise ValueError("ACT n_action_steps must be an integer between 1 and chunk_size")
                self.temporal_agg = False
                default_horizon = horizon
                print(f"ACT inference: n_action_steps={horizon}, temporal_agg=False, "
                      "use_cuda_graph=False", flush=True)
        elif self.kind == "n0vtla":
            from transformers import AutoTokenizer
            from .n0vtla.configuration_n0vtla import N0PolicyConfig
            from .n0vtla.modeling_n0vtla import N0Policy
            config = N0PolicyConfig(**self.run["config"]["n0_config"])
            config.use_cuda_graph = args.get("use_cuda_graph", True)
            if "num_inference_steps" in args:
                value = args["num_inference_steps"]
                if type(value) is not int or value < 1:
                    raise ValueError("num_inference_steps must be a positive integer")
                config.num_inference_steps = value
            if "n_action_steps" in args:
                value = args["n_action_steps"]
                if type(value) is not int or not 1 <= value <= config.chunk_size:
                    raise ValueError("n_action_steps must be an integer between 1 and chunk_size")
                config.n_action_steps = value
            self.tokenizer = AutoTokenizer.from_pretrained(asset_path(self.run, "tokenizer_path"), local_files_only=True)
            self.model = N0Policy(config).to(self.device)
            print(f"N0VTLA inference: n_action_steps={config.n_action_steps}, "
                  f"use_cuda_graph={config.use_cuda_graph}", flush=True)
            self.chunk_size, self.temporal_agg = config.chunk_size, False
            default_horizon = config.n_action_steps
        else:
            config = make_vla_config(self.run, self.device, models_root)
            if self.kind == "starvla_pi":
                from policy.starvla_pi.modeling_starvla_pi import StarvlaPiPolicy
                for name in ("n_action_steps", "num_inference_timesteps"):
                    if name in args:
                        value = args[name]
                        if type(value) is not int or value < 1 or (name == "n_action_steps" and value > config.chunk_size):
                            raise ValueError(f"Invalid {name}")
                        setattr(config, name, value)
                config.use_cuda_graph = args.get("use_cuda_graph", config.use_cuda_graph)
                self.model = StarvlaPiPolicy(config).to(self.device)
                print(f"StarVLA-pi inference: n_action_steps={config.n_action_steps}, "
                      f"num_inference_timesteps={config.num_inference_timesteps}, "
                      f"chunk_size={config.chunk_size}, expert_width={self.model.action_head.config.hidden_size}, "
                      f"use_cuda_graph={config.use_cuda_graph}", flush=True)
            elif self.kind == "starvla_groot":
                from policy.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy
                config.use_cuda_graph = args.get("use_cuda_graph", config.use_cuda_graph)
                for name in ("n_action_steps", "num_inference_timesteps"):
                    if name in args:
                        value = args[name]
                        if type(value) is not int or value < 1:
                            raise ValueError(f"{name} must be a positive integer")
                        if name == "n_action_steps" and value > config.chunk_size:
                            raise ValueError("n_action_steps must not exceed chunk_size")
                        setattr(config, name, value)
                config.repeated_diffusion_steps = args.get("repeated_diffusion_steps", config.repeated_diffusion_steps)
                if not isinstance(config.repeated_diffusion_steps, int) or config.repeated_diffusion_steps < 1:
                    raise ValueError("repeated_diffusion_steps must be a positive integer")
                self.model = StarvlaGrootPolicy(config).to(self.device)
                print(f"StarVLA inference: n_action_steps={config.n_action_steps}, "
                      f"num_inference_timesteps={config.num_inference_timesteps}, "
                      f"use_cuda_graph={config.use_cuda_graph}", flush=True)
            else:
                from transformers import AutoTokenizer
                from policy.pi05.modeling_pi05 import PI05Policy
                self.tokenizer = AutoTokenizer.from_pretrained(config.paligemma_tokenizer_path, local_files_only=True)
                config.use_cuda_graph = args.get("use_cuda_graph", True)
                if "num_inference_steps" in args:
                    value = args["num_inference_steps"]
                    if type(value) is not int or value < 1:
                        raise ValueError("num_inference_steps must be a positive integer")
                    config.num_inference_steps = value
                if "n_action_steps" in args:
                    value = args["n_action_steps"]
                    if type(value) is not int or not 1 <= value <= config.chunk_size:
                        raise ValueError("n_action_steps must be an integer between 1 and chunk_size")
                    config.n_action_steps = value
                self.model = PI05Policy(config).to(self.device)
                print(f"PI05 inference: n_action_steps={config.n_action_steps}, "
                      f"use_cuda_graph={config.use_cuda_graph}", flush=True)
            self.chunk_size = config.chunk_size
            self.temporal_agg = False
            default_horizon = config.n_action_steps

        if "chunk_size" in args and args["chunk_size"] != self.chunk_size:
            raise ValueError("Requested chunk_size does not match checkpoint architecture")

        # mmap keeps the optimizer tensors in the training bundle out of RAM.
        payload = torch.load(self.run["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        require_rgb_model(payload)
        if "model" not in payload:
            raise ValueError("Expected a train_policy.sh checkpoint containing a 'model' state dict")
        if self.run["step"] is not None and payload.get("step") != self.run["step"]:
            raise ValueError("Checkpoint filename step does not match its saved training step")
        if payload.get("policy", self.kind) != self.kind:
            raise ValueError("Checkpoint policy does not match train_config.yml")
        if payload.get("action_mode", "absolute_joint") != self.action_mode:
            raise ValueError("Checkpoint action_mode does not match train_config.yml")
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
        if self.kind in ("pi05", "n0vtla", "starvla_pi"):
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
            image = observation["observation"][camera]["rgb"]
            if self.kind == "n0vtla":
                from .n0vtla.observation_adapter import resize_rgb
                batch[key] = resize_rgb(rgb_image(image)).unsqueeze(0)
            else:
                batch[key] = rgb_tensor(image, self.image_size).unsqueeze(0)
        if self.tactile_mode != "none":
            for side in ("left", "right"):
                stream = observation["tactile"][f"{side}_tactile"]
                if self.tactile_input_mode in ("rgb_only", "marker_rgb"):
                    key = "rgb" if self.tactile_input_mode == "rgb_only" else "rgb_marker"
                    raw = {key: rgb_image(stream[key])}
                else:
                    raw = {"marker": as_numpy(stream["marker"])}
                    if self.tactile_input_mode in DEPTH_DEFORM_SPECS:
                        raw["depth"] = as_numpy(stream["depth"])
                        for field in ("press_depth", "marker_full"):
                            if field in stream:
                                raw[field] = as_numpy(stream[field])
                if self.tactile_mode == "native":
                    from .n0vtla.observation_adapter import native_tactile_from_arrays
                    tactile = native_tactile_from_arrays(raw, self.tactile_input_mode)
                    if side not in self.tactile_baselines:
                        baseline = stream.get("baseline")
                        if baseline is not None:
                            if self.tactile_input_mode in ("rgb_only", "marker_rgb"):
                                baseline_raw = {key: rgb_image(baseline[key])}
                            else:
                                baseline_raw = {k: as_numpy(v) for k, v in baseline.items()}
                        elif self.tactile_input_mode == "rgb_only" and "baseline_rgb" in stream:
                            baseline_raw = {"rgb": rgb_image(stream["baseline_rgb"])}
                        else:
                            baseline_raw = raw
                        self.tactile_baselines[side] = native_tactile_from_arrays(baseline_raw, self.tactile_input_mode)
                    batch[f"observation.images.tac_{side}"] = tactile.unsqueeze(0)
                    batch[f"observation.images.tac_{side}.baseline"] = self.tactile_baselines[side].unsqueeze(0)
                    continue
                tactile = tactile_from_arrays(raw, self.tactile_input_mode, self.image_size)
                if getattr(self, 'encoder_pretrained_mode', 'univtac') == 'tactile' and self.tactile_input_mode in ('rgb_only', 'marker_rgb'):
                    if side not in self.tactile_baselines:
                        self.tactile_baselines[side] = tactile.clone()
                    tactile = (tactile - self.tactile_baselines[side] + self.tactile_background_offset).clamp(0, 1)
                if getattr(self, 'tactile_num_frames', 1) > 1:
                    simulation_stride = getattr(self, 'tactile_simulation_step_stride', None)
                    if simulation_stride is not None:
                        from encoder.temporal_sampling import SimulationTactileHistory
                        if 'physics_step' not in observation:
                            raise ValueError('Temporal encoder inference requires an explicit physics_step counter')
                        if side not in self.tactile_history:
                            self.tactile_history[side] = SimulationTactileHistory(self.tactile_num_frames, simulation_stride)
                        history = self.tactile_history[side]
                        history.append(tactile, observation['physics_step'])
                        tactile = history.clip(require_complete=not getattr(self, '_warming_tactile_history', False))
                    else:
                        depth = (self.tactile_num_frames - 1) * self.tactile_frame_stride + 1
                        if side not in self.tactile_history:
                            self.tactile_history[side] = deque([tactile] * depth, maxlen=depth)
                        self.tactile_history[side].append(tactile)
                        from encoder.pretrained_registry import history_indices
                        tactile = torch.stack([self.tactile_history[side][i]
                            for i in history_indices(depth - 1, self.tactile_num_frames, self.tactile_frame_stride)])
                batch[f"observation.images.tac_{side}"] = tactile.unsqueeze(0)
        if self.tokenizer is not None:
            if self.kind == "n0vtla":
                from .n0vtla.observation_adapter import tokenize_batch
                tokenize_batch(batch, self.tokenizer, self.model.config.tokenizer_max_length)
            else:
                from .pi05.tokenization import make_pi_tokens
                make_pi_tokens(batch, self.tokenizer, self.model.config.tokenizer_max_length)
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()}

    @torch.no_grad()
    def predict_chunk(self, observation, batch=None):
        batch = self.encode_obs(observation) if batch is None else batch
        if self.record_policy_video or self.save_policy_inputs:
            self.last_video_images = {key: value.detach().cpu().clone() for key, value in batch.items()
                                      if key.startswith("observation.images.")}
            self.last_video_input_step = self.time
            if self.save_policy_inputs:
                from .input_recording import snapshot_observation
                self.last_video_observation = snapshot_observation(observation)
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
        if self.kind in ("pi05", "n0vtla", "starvla_pi"):
            low, high = normalized.new_tensor(stats["q01"]), normalized.new_tensor(stats["q99"])
            actions = (normalized + 1) * (high - low).clamp_min(1e-6) / 2 + low
        else:
            actions = normalized * normalized.new_tensor(stats["std"]) + normalized.new_tensor(stats["mean"])
        anchor = actions.new_tensor(as_numpy(observation["embodiment"]["joint"]).reshape(-1)[:8])
        return decode_joint_actions(anchor, actions[0].float(), self.action_mode)

    def eval(self, task, observation):
        # Advance tactile history at EVERY environment step, including queued actions.
        if getattr(self, 'tactile_simulation_step_stride', None) is not None:
            if int(task.cfg.decimation) != self.tactile_simulation_step_stride:
                raise ValueError('Inference must observe every 2 physical simulation steps; '
                                 'configure eval decimation=2 to match encoder training')
            # Gather actual observations while holding the current command before
            # the first policy action. Repeating the first frame would differ from
            # the complete-history windows used in encoder and policy training.
            self._warming_tactile_history = True
            try:
                batch = self.encode_obs(observation)
                while not all(history.ready for history in self.tactile_history.values()):
                    previous_physics_step = int(observation['physics_step'])
                    with task._configured_decimation():
                        task._step(is_save=False)
                    observation = task._get_observations()
                    if int(observation['physics_step']) - previous_physics_step != self.tactile_simulation_step_stride:
                        raise ValueError('Tactile warmup did not advance exactly two physical simulation steps')
                    batch = self.encode_obs(observation)
            finally:
                self._warming_tactile_history = False
        else:
            batch = self.encode_obs(observation) if getattr(self, 'tactile_num_frames', 1) > 1 else None
        if self.kind == "act" and self.temporal_agg:
            self.history.append((self.time, self.predict_chunk(observation, batch)))
            self.history = [(start, chunk) for start, chunk in self.history if self.time - start < len(chunk)]
            candidates = torch.stack([chunk[self.time - start] for start, chunk in self.history])
            weights = torch.exp(-0.01 * torch.arange(len(candidates), device=candidates.device))
            action = (candidates * (weights / weights.sum())[:, None]).sum(0)
        else:
            if not self.actions:
                self.actions.extend(self.predict_chunk(observation, batch)[:self.execution_horizon])
            action = self.actions.popleft()
        recorder = getattr(task, "policy_input_recorder", None)
        if recorder is not None:
            recorder.write(task, self.last_video_images, self.last_video_observation,
                           self.time, self.last_video_input_step)
            recorder.write_markers(task, observation, self.time, self.last_video_input_step)
        self.time += 1
        return task.take_action(action.to(task.device), action_type="qpos")

    def reset(self):
        self.last_video_images = None
        self.last_video_observation = None
        self.last_video_input_step = None
        self.tactile_baselines = {}
        self.tactile_history = {}
        self._warming_tactile_history = False
        self.actions = deque()
        self.history = []
        self.time = 0
        if hasattr(self.model, "reset"):
            self.model.reset()
