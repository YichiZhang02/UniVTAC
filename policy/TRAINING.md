# Policy training on UniVTAC demonstrations

Use `ENCODER_PRETRAINED_MODE=univtac|tactile|vision` together with `ENCODER_METHOD`
for encoded tactile inputs. The launcher selects the smallest configured official
variant and fine-tunes the encoder with the policy. Temporal encoders use four
frames at stride two; Wan2.2-VAE uses five chronological frames at stride two and
outputs 98×192 patchified features per sensor. See [encoder sources and examples](../encoder/PRETRAINED.md)
for downloads, checkpoint migration, supported sizes and shared adaptation parameters.

Set `ENCODER_INIT=official` with `ENCODER_PRETRAINED_MODE=tactile` or
`ENCODER_PRETRAINED_MODE=vision ENCODER_METHOD=Wan2.2-VAE`, and no `ENCODER_CKPT`,
to initialize downstream policy training directly from the official weights,
skipping discovery of existing encoder runs. Rank zero only
converts the weights into the policy checkpoint format; it performs no separate
encoder optimization. New Sparsh initializers use 224×224 inputs. The default
`ENCODER_INIT=latest` retains automatic selection of the latest exported encoder.

`train_policy.sh` trains ACT, PI0.5, StarVLA-GR00T, or N0-VTLA from the same raw Isaac 5.1 HDF5 files. All training code and model assets are local to this repository. The state is `embodiment/joint[t, :8]`; the action chunk begins with `embodiment/joint[t+1, :8]`. Those eight values are seven arm joints and one gripper value. The ninth raw joint column is excluded, matching ACT.

## Setup

Follow the training-environment Quick Start in [README_yichi.md](../README_yichi.md), then install `requirements_train.txt` in the activated Python 3.11 environment. This single file includes the pinned PyTorch 2.7.0 / torchvision 0.22.0 CUDA 12.6 builds, training dependencies, and the Qwen3.5 CUDA extension. PI0.5 weights are expected in `resources/pretrained_models/pi05_base/`; Qwen3.5-2B in `resources/pretrained_models/Qwen3.5-2B/`. Both directories are Git ignored. Check that `torch.cuda.is_available()` is true before a production run.

The included `causal_conv1d` prebuilt wheel targets Linux x86_64, Python 3.11, Torch 2.7, CUDA 12, and the C++11 ABI. Update the Torch pins, wheel source, and extension URL together when changing that stack. These instructions cover the training environment; simulator testing and evaluation environments remain unverified.

## Examples

```bash
# One ACT model for one task, without tactile input.
POLICY=act TRAINING_MODE=single_task TASK=lift_can TACTILE_MODE=none bash train_policy.sh

# One PI0.5 model trained on all available tasks, tactile as images.
POLICY=pi05 TRAINING_MODE=multi_task EPISODES=0 \
  TACTILE_MODE=as_image TACTILE_INPUT_MODE=marker_rgb bash train_policy.sh

# One StarVLA-GR00T model, with the local MAE-B depth/deformation encoder.
POLICY=starvla_groot TRAINING_MODE=multi_task \
  TACTILE_MODE=encode TACTILE_INPUT_MODE=depth_deform_80 TACTILE_TYPE=full \
  ENCODER_METHOD=MAE ENCODER_SIZE=B bash train_policy.sh
```

`TRAINING_MODE=single_task` uses `TASK`; `multi_task` uses all available task directories by default, or the comma-separated `TASKS` subset. `EPISODES` limits episodes **per task**; `0` means all. Multi-task sampling balances tasks so that a longer task does not dominate training. The train/validation split is by complete episode, with seed 42 by default.

`TACTILE_MODE` is `none`, `as_image`, or `encode`. `TACTILE_INPUT_MODE` is `marker_only`, `rgb_only`, `marker_rgb`, or `depth_deform_20`, `depth_deform_80`, `depth_deform_160`, `depth_deform_240`, `depth_deform_nonlinear`, `depth_deform_nonlinear_radial`; `INPUT_MODE` is accepted as a compatibility alias. `encode` additionally uses `TACTILE_TYPE=cls|full|pool` and `ENCODER_CKPT`, or the latest matching `encoder_results/<input>/<pretrained_mode>/<method>/<size>/*/encoder.pth`. `as_image` sends the selected tactile representation through a visual path. All depth/deform modes use `clip((30-camera_depth_mm)/5,0,1)` from raw camera depth. Linear modes use ±20/80/160/240 px; their previous 28 mm / 3 mm checkpoints need retraining. `depth_deform_nonlinear` uses linear displacement mapping within ±20 px with rational tails, and TPS interpolation; `depth_deform_nonlinear_radial` applies the original nonlinear amplitude curve to vector length r=sqrt(dx^2+dy^2), scaling dx and dy together to preserve direction; the linear core is r<=20 px and rational tails have no hard cutoff; both use full-marker TPS. See `encoder/README.md` for formulas. Testing retains out-of-view marker coordinates, stable IDs and visibility in `marker_traces/<seed>_<result>.hdf5`. `marker_only` is already three-channel RGB (`R = G = B`, white marker dots on black), shared by the `as_image` and `encode` paths. Older one-channel marker encoder checkpoints require retraining or explicit weight conversion. Set `CAMERAS=head|all|auto`; `auto` uses both cameras only when every selected task is configured for both.

Set `TACTILE_TYPE=pool` (Python: `--tactile-type pool`) with `TACTILE_MODE=encode` to apply policy-side adaptive average pooling to the tactile encoder's spatial output before the policy projection and VLM/action head. Supported by ACT, PI0.5, StarVLA-GR00T, and StarVLA-PI. Each spatial grid becomes 3×3 (9 tokens); temporal slices and sensors are pooled separately. All six UniVTAC methods prepend their `cls`-mode feature to the nine spatial tokens, giving 10 tokens per image (20 for two sensors). ResNet uses its CNN global feature in place of CLS and applies its pretrained FC to the pooled spatial features to match the 512-D global feature; V-JEPA applies its projector to both CLS and all patch tokens before pooling. Other ViT methods use the final normalized CLS. `full` and `cls` retain their original behavior. The type is saved in the policy run configuration and restored automatically at inference. Existing checkpoints retain their saved type and state-dict keys; no pooling is applied to old `full`/`cls` runs. Encoder pretraining and N0-VTLA's native tactile path are unchanged.

The per-policy `base_config.yml` files hold model and optimizer defaults. Routing, task selection, data representation, training budget, and output destination are set in `train_policy.sh` using environment variables. PI0.5 and GR00T retain the StarVTLA optimizer defaults, including AdamW at `2.5e-5` and a warmup followed by cosine decay. The root launcher and `training/*.sh` default to `CHUNK_SIZE=50` and `N_ACTION_STEPS=30`. The root launcher accepts environment overrides; `training/*.sh` fixes these two values with explicit exports. Require `1 <= N_ACTION_STEPS <= CHUNK_SIZE`. The execution horizon is saved with the run and becomes the deployment default; `--execution-horizon` can override it at inference. ACT with temporal aggregation enabled still replans every step. All three policies use an **absolute 8D joint state**, with the first action target at `t+1`. Action targets support the two modes below.

### Absolute and relative joint actions

Set `ACTION_MODE=absolute` or `ACTION_MODE=relative`; the canonical names `absolute_joint` and `relative_joint` are also accepted. ACT, PI0.5 and GR00T default to absolute; N0-VTLA defaults to relative in the Bash launcher. Direct Python invocations use `--action-mode` (default absolute). All four policies support both modes through the unified entry points.

| Mode | Arm target at chunk index k | Gripper target |
| --- | --- | --- |
| `absolute_joint` | `joint[t+1+k, :7]` | `joint[t+1+k, 7]` |
| `relative_joint` | `joint[t+1+k, :7] - joint[t, :7]` | `joint[t+1+k, 7]` |

Every relative arm target in a chunk uses the **same current observation** as its anchor. These are position differences, not velocities or consecutive-frame differences. Arm values retain the joint angle units; the gripper retains its raw absolute joint position units. The state input stays absolute in both modes.

```bash
POLICY=pi05 TRAINING_MODE=single_task TASK=lift_can TACTILE_MODE=none \
  ACTION_MODE=relative bash train_policy.sh

POLICY=starvla_groot TRAINING_MODE=multi_task TACTILE_MODE=none \
  ACTION_MODE=absolute bash train_policy.sh
```

Training converts targets before normalization and computes new action statistics from the training episodes. Relative action moments cover every valid target in the configured chunk horizon, excluding padded tail targets; quantiles use a deterministic uniform sample of at most 200,000 targets to bound memory. Absolute mode preserves the existing next-frame statistics. PI0.5 uses quantile normalization, while ACT and GR00T use mean/std normalization. Changing action mode or chunk horizon requires a new training run and its own statistics.

Deployment reads the mode from the saved run, reverses normalization, and adds the original observation's arm position back to the entire relative chunk. The resulting absolute targets enter the execution queue or ACT temporal aggregation, then the simulator's existing `qpos` API. The gripper is never offset. Do not manually override a checkpoint's action mode at evaluation time.

The mode is saved in `train_config.yml`, `dataset_stats.json`, and each checkpoint payload; inconsistent modes are rejected on load. Older checkpoints without action-mode metadata default to `absolute_joint`. Relative run labels include `_relative_joint` after the tactile mode; existing absolute run names are unchanged. This change covers the unified trainer/deployment bridge; historical standalone policy training scripts retain their own conventions.

### N0-VTLA native tactile backend

```bash
POLICY=n0vtla TRAINING_MODE=single_task TASK=lift_can \
  ACTION_MODE=relative BATCH_SIZE=8 bash train_policy.sh
```

N0 defaults to `TACTILE_MODE=native`, `TACTILE_INPUT_MODE=rgb_only`, chunk 50 and 20,000 steps. All nine tactile input representations (`rgb_only`, `marker_rgb`, `marker_only`, `depth_deform_20`, `depth_deform_80`, `depth_deform_160`, `depth_deform_240`, `depth_deform_nonlinear`, `depth_deform_nonlinear_radial`) are supported. Current and episode-first frames are converted to the selected representation before subtraction, then use the original frozen-DINOv2/predictor/latent-token pathway. `TACTILE_MODE=none` provides a missing-touch control with the same model structure. N0 uses quantile normalization and masks padded action time steps and the 24 unused action dimensions in its loss.

The base checkpoint is loaded strictly from `resources/pretrained_models/n0-vtla-base/` (`N0_PRETRAINED_PATH` override); the learned gate and tactile parameters are retained. The existing local PaliGemma tokenizer is bundled into the saved run (`N0_TOKENIZER_PATH` override). The saved model contains DINOv2 weights, so inference requires neither the initialization checkpoint nor a backbone download. Runtime imports are entirely inside UniVTAC; the adjacent N0-VTLA repository is not required. No Transformers installation files are overwritten. See [N0-VTLA details](n0vtla/README.md).

StarVLA-GR00T and StarVLA-PI now default to `action_expert_hidden_size: 1024` in their base configs. This is the transformer width, separate from GR00T's `action_head_hidden_size` MLP width. Qwen3.5-2B remains 24 text layers at width 2048; PI uses 24 action blocks with cross-attention projections connecting the two widths. Both action heads use FP32 autocast for training and inference, with TF32 disabled. Old run/config files lacking the expert-width field restore the historical architecture (GR00T's DiT-B/L preset, or PI's VLM width), including old PI selected-layer layouts, and load weights strictly. New runs save the width explicitly. Raw state dictionaries still require a matching architecture config; weights are never resized to the new defaults.

Each run saves `train.log`, `exit_code`, `train_config.yml`, `dataset_stats.json`, `episodes.json`, and step checkpoints under `policy_results/` unless `OUTPUT_DIR` is set. No simulator is started by this training entry point.

All policy implementations live under `policy/`: PI0.5 is in `pi05/`, StarVLA-GR00T in `starvla_groot/`, and their shared infrastructure in `vla_common/`. Each VLA policy keeps its own `base_config.yml` beside its model implementation. The unified trainer and deployment bridge use `policy.pi05` / `policy.starvla_groot` and the same shared config classes.

The public root Bash entry points are `train_encoder.sh`, `train_policy.sh`, and `test_policy.sh`. Data collection, parallel evaluation, and experiment-grid helpers live in `scripts/`. Helpers launch the same root training/testing entry points where applicable.

## Unified checkpoint names and testing

New policy run directories use `<timestamp>_<policy>_<training_mode>_<task_label>_<tactile_mode>` followed by the tactile input when enabled, and encoder method, size, token type, and insertion location in `encode` mode. For example: `20261004-120000_starvla_groot_multi_task_all_encode_marker_rgb_ResNet_S_cls_encoder`. The four node launchers use the same convention. Each checkpoint remains `checkpoint_<step>.pt`, containing the model state and optimizer/scheduler state for that training step. Run directories and `checkpoint_<step>.pt` are actual directories/files. Training does not create a `last.pt` symlink; testing a run directory selects the greatest available checkpoint step.

Training output is recorded in each run directory: rank 0 writes `train.log` and the other DDP ranks write `train_rank<N>.log`. The grid/node launchers also record `launcher.log` and the process exit status in `exit_code`. Single-GPU grid scheduling logs are under the corresponding suite's `logs/` subdirectory. Existing 24 experiments' logs and exit statuses have been moved into their result directories.

Existing completed runs have been moved into canonical physical directories directly under `policy_results/`, indexed in `policy_results/checkpoints_index.json`. Checkpoint contents are preserved by filesystem renames. The former directory aliases and `last.pt` symlinks have been removed. `python scripts/organize_policy_checkpoints.py --apply` organizes runs into this layout and updates the index; without `--apply`, it only prints the plan. Transfer an experiment directory over SSH with `rsync -a`, keeping its configuration/statistics beside the checkpoint.

```bash
RUN_DIR=/path/to/policy_results/<canonical_run_name>

# Inspect an actual checkpoint file without loading weights or starting Isaac Sim.
bash test_policy.sh --checkpoint "$RUN_DIR/checkpoint_30000.pt" --inspect

# Load the latest checkpoint and infer one recorded HDF5 frame; no simulator.
bash test_policy.sh --checkpoint "$RUN_DIR" --task lift_can --offline

# Choose a particular training step.
bash test_policy.sh --checkpoint "$RUN_DIR" --step 10000 --task lift_can --offline

# Simulator evaluation; the 5090 environment remains to be validated.
bash test_policy.sh --checkpoint "$RUN_DIR" --task lift_can --task-config demo --gpu 0 --headless

# Legacy positional baseline deploy configs use the Python entry point directly.
python scripts/test_policy.py lift_can demo ACT/deploy 0
```

### Parallel multi-task simulator evaluation

The first four positional arguments to `test_policy.sh` are model ID, step, GPU IDs, and GPU count. Each uses the default shown in the script when omitted (`${1:-...}` and so on), so `bash test_policy.sh` launches the default run. The other batch defaults (episode count, seeds, and reset timeout) are at the top of the script. A local `../miniconda3/envs/univtac/bin/python` is selected automatically when present. `bash test_policy.sh --dry-run` prints the default plan without starting Isaac Sim.

Pass the directory name under `policy_results/` and a checkpoint step. List only GPUs available for this evaluation in the third argument; the fourth argument uses the first N listed GPUs, with one task process per GPU. As a task finishes, its GPU runs the next pending task. The default task list comes from `episodes.json` when present. For older `multi_task` exports without that file, it is the eight simulator tasks with instruction files.

```bash
bash test_policy.sh

# If GPUs 1, 4 and 6 are occupied, use the remaining GPUs with three workers.
bash test_policy.sh \
  20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder \
  30000 0,2,5 3

# Inspect selected tasks and settings without launching the simulator.
bash test_policy.sh --dry-run
```

Batch defaults from `test_policy.sh` are `demo` task config, 20 valid episodes per task, start seed `-1` (resolved to `1000000` for checkpoint seed 0), no maximum seed, and `seen` instructions. Batch execution is headless with livestream disabled and a 600-second reset limit. It stops a task after 10 evaluation exceptions to avoid endless retries; `--max-errors` changes this cap. `--total-num`, `--start-seed`, `--max-seed`, `--tasks`, and simulator `--config-overrides` are available.

Outputs are in `test_results/<model_id>/<step>/`: `logs/<task>.log`, `videos/<task>/video/<seed>.mp4`, `results/<task>.json`, `settings.json`, `summary.json`, and `summary.md`. The average in the summary is the arithmetic mean of the selected task success rates. It is shown only when every selected task reaches the requested number of valid episodes. An existing output directory is renamed to a sibling ending in `_previous_<timestamp>` before a new run, preserving interrupted results without mixing them.

`--checkpoint` accepts a physical run directory or a particular `checkpoint_<step>.pt` file; directories select the greatest available step. `CHECKPOINT=...` can be used instead of the flag. Keep `train_config.yml`, `dataset_stats.json`, and `episodes.json` beside the weights when transferring a run. `--models-root` overrides the local `resources/pretrained_models/` directory. The unified loader restores model architecture, camera routing, tactile representation, state normalization, and action denormalization from the saved training metadata. VLA tactile backbones are reconstructed from saved architecture metadata and restored from the policy state, so they do not depend on the original server's absolute encoder path. Language conditioning uses the same task-name labels used in training.

`--inspect` is a metadata check; `--offline` checks checkpoint loading and action inference. Neither reports simulator task success. A 5090 smoke evaluation of the 30,000-step StarVLA-GR00T `lift_can` checkpoint completed one simulator episode (seed 0, 0/1 success); see [README_5090.md](../resources/docs/README_5090.md). `scripts/eval_policy.sh` is a compatibility wrapper that forwards to `test_policy.sh`; `scripts/parallel_eval.sh` retains parallel evaluation with an explicit deploy configuration. The shell script's positional arguments now select a batch model, so legacy positional task/deploy arguments must go to `scripts/test_policy.py`.

## Distributed training and the 32-GPU experiment grid

`NUM_PROCESSES=4 bash train_policy.sh` launches one experiment with four DDP ranks using `torchrun`. Select its GPUs with `CUDA_VISIBLE_DEVICES=0,1,2,3`. `BATCH_SIZE` is the **global** batch size and must be divisible by the number of ranks: `BATCH_SIZE=64 NUM_PROCESSES=4` gives 16 samples per GPU. `WORKERS` is the number of DataLoader CPU workers **per rank**, not the GPU count. A training step is one synchronized optimizer update across all ranks. Checkpoints and validation are handled by rank 0; checkpoint model keys have no DDP `module.` prefix. Multi-task sampling retains task balancing, sharding one deterministic weighted draw across ranks and refreshing it when the loader completes an epoch.

For `depth_deform_nonlinear` and `depth_deform_nonlinear_radial`, `TACTILE_PREPROCESSING=auto` (the default) moves dense TPS evaluation to CUDA during training. Data workers solve the small TPS systems and load camera depth; the GPU constructs the same dense displacement fields in bounded frame batches. This avoids rebuilding a dense CPU interpolation map for almost every shuffled tactile frame. Duplicate-marker merging, camera-depth normalization, temporal windows, and displacement encoding retain the CPU reference semantics. Other representations use their existing preprocessing. Set `TACTILE_PREPROCESSING=cpu` to run the reference path. `data_wait_max_ms` reports the greatest data wait across ranks, since a slow nonzero rank can stall the entire update while rank 0 reports little local waiting.

For the two StarVLA Wan2.2-VAE encode commands, `WAN_TRAINING_OPTIMIZATION=compiled` opts into full-clip five-frame encoding, channels-last 3D convolution weights, CuDNN algorithm selection, and compilation of the encoder forward/backward. Five frames and all trainable encoder parameters are retained. The reference path remains the default; set the flag only on a Wan command. Parameter names, shapes and dtypes are checkpoint compatible. Eval/no-grad, non-five-frame, tiled and sliced execution use the streaming reference. Compilation/autotuning adds first-step startup time and can produce ordinary floating-point differences; it does not promise bitwise identical training. With the stage profiler, add `--wan-training-optimization compiled`. `scripts/test_wan_training_fast.py` checks temporal boundaries, gradients, fallback paths and state-dict compatibility.

To measure training stages on an existing run's episode split and encoder initializer:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NCCL_DEBUG=WARN \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/profile_starvla_run.py \
  --run-config policy_results/<run>/train_config.yml \
  --output /tmp/starvla_stage_profile --batch-per-gpu 16 --steps 25 --warmup 5
```

This uses real HDF5 observations, verifies and reuses the saved dataset statistics, and starts from the original pretrained weights. It does not resume or overwrite the source run. `--preprocessing cpu` measures the CPU reference. Each measured step synchronizes CUDA so its stage timings add up to wall time; profiling can modestly affect throughput. The output contains per-rank samples and mean/median/p90 summaries in `stage_profile.json`. Forward timings for the tactile encoder, Qwen, and action head are subdivisions of `forward_ms`; Qwen input preparation is itself a subdivision of Qwen time. `backward_ddp_ms` includes overlapping backward computation, gradient communication, and rank waits, and cannot be interpreted as pure communication time. Initialization, warmup, validation, and checkpoint writes are excluded from the steady-step summaries. The ordinary trainer also accepts `--profile-stages --profile-warmup 5`.

Run the following four scripts on four separate eight-GPU machines, one script per machine. Each script launches two independent four-GPU DDP experiments on GPU groups `0,1,2,3` and `4,5,6,7`, running six methods in three rounds. Across the four machines, up to eight experiments run concurrently; the 24 configurations do not overlap.

| Machine | Script | Tactile input | Tactile type |
| --- | --- | --- | --- |
| 1 | `scripts/train_starvla_groot_node1.sh` | `marker_rgb` | `cls` |
| 2 | `scripts/train_starvla_groot_node2.sh` | `marker_rgb` | `full` |
| 3 | `scripts/train_starvla_groot_node3.sh` | `depth_deform_20`, `depth_deform_80`, `depth_deform_160`, `depth_deform_240` | `cls` |
| 4 | `scripts/train_starvla_groot_node4.sh` | `depth_deform_20`, `depth_deform_80`, `depth_deform_160`, `depth_deform_240` | `full` |

Every script uses all available tasks and all episodes (with the existing episode-level training/validation split), head and wrist cameras, tactile encoding inserted at `encoder`, six encoder methods (`ResNet`, `VAE`, `MAE`, `DINOv2`, `I-JEPA`, `V-JEPA`), size S, the fixed trained checkpoints listed in `scripts/train_starvla_groot_node.sh`, 30,000 optimizer steps, global batch 64, chunk 32, and a checkpoint every 5,000 steps. Both concurrent experiments use their own automatic rendezvous port. Each method has a separate result directory containing `train.log`, `launcher.log`, and `exit_code`; DDP ranks beyond rank 0 write `train_rank<N>.log` in that directory; failed experiments are reported while remaining methods continue.

```bash
# On machine 1; use the corresponding node2/node3/node4 script on the other machines.
bash scripts/train_starvla_groot_node1.sh --dry-run
bash scripts/train_starvla_groot_node1.sh

# Override the Python path if the Conda environment lives elsewhere.
PYTHON_BIN=/path/to/univtac/bin/python bash scripts/train_starvla_groot_node1.sh
```

Each machine needs the repository, demonstration data, trained encoder checkpoints, and `resources/pretrained_models/Qwen3.5-2B/` at the same relative locations. `GPUS`, `OUTPUT_ROOT`, `RUN_ID`, and `WORKERS` can be overridden through environment variables. The earlier `scripts/train_starvla_groot_grid.sh` remains a launcher for independent single-GPU experiments; use the four node scripts for this DDP layout.

## Local code provenance

`policy/vla_common/engine` and the PI0.5 and StarVLA-GR00T model implementations were adapted from the adjacent StarVTLA checkout, including its attributed LeRobot and OpenPI-derived source. The training data adapter and tactile checkpoint adapter are UniVTAC-specific. Runtime imports and weights resolve entirely within this repository.

## StarVLA-PI

`POLICY=starvla_pi` 使用 `policy/starvla_pi/base_config.yml`，与 StarVLA-GR00T 共用
`resources/pretrained_models/Qwen3.5-2B/` 预训练 VLM；动作头随机初始化。
支持相同的 `none/as_image/encode` 触觉路由和 `encoder/decoder` 插入位置。
State/action 使用 q01/q99 分位数归一化；真实 8D 动作归一化后补齐到 32D，
补齐维度及 episode 尾部无效时间步不参与 loss，部署输出裁回 8D 后反归一化。
默认使用全部 24 层 Qwen hidden states、24 个对应 DiT block 和 10 步 Euler 去噪。
推理默认开启 CUDA Graph，捕获完整逐层 VLM 和动作去噪，使用单次噪声采样。

```bash
POLICY=starvla_pi TRAINING_MODE=single_task TASK=lift_can EPISODES=2 \
  TACTILE_MODE=none CAMERAS=all STEPS=1 BATCH_SIZE=1 WORKERS=0 \
  MAX_VAL_BATCHES=1 bash train_policy.sh

POLICY=starvla_pi TRAINING_MODE=multi_task CAMERAS=all \
  TACTILE_MODE=encode TACTILE_INPUT_MODE=marker_rgb \
  ENCODER_METHOD=VAE ENCODER_SIZE=S TACTILE_TYPE=full \
  TACTILE_INSERT_LOCATION=encoder bash train_policy.sh
```

该动作头宽度与 Qwen hidden size 一致，显存需求与 GR00T 不同；先用小 batch 测量。
训练 checkpoint 和 `test_policy.sh` 部署入口沿用统一格式，详见
[StarVLA-PI README](starvla_pi/README.md)。
