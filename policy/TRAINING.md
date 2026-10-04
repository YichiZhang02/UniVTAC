# Policy training on UniVTAC demonstrations

`train_policy.sh` trains ACT, PI0.5, or StarVLA-GR00T from the same raw Isaac 5.1 HDF5 files. All training code and model assets are local to this repository. The state is `embodiment/joint[t, :8]`; the action chunk begins with `embodiment/joint[t+1, :8]`. Those eight values are seven arm joints and one gripper value. The ninth raw joint column is excluded, matching ACT.

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
  TACTILE_MODE=encode TACTILE_INPUT_MODE=depth_deform TACTILE_TYPE=full \
  ENCODER_METHOD=MAE ENCODER_SIZE=B bash train_policy.sh
```

`TRAINING_MODE=single_task` uses `TASK`; `multi_task` uses all available task directories by default, or the comma-separated `TASKS` subset. `EPISODES` limits episodes **per task**; `0` means all. Multi-task sampling balances tasks so that a longer task does not dominate training. The train/validation split is by complete episode, with seed 42 by default.

`TACTILE_MODE` is `none`, `as_image`, or `encode`. `TACTILE_INPUT_MODE` is `marker_only`, `rgb_only`, `marker_rgb`, or `depth_deform`; `INPUT_MODE` is accepted as a compatibility alias. `encode` additionally uses `TACTILE_TYPE=cls|full` and `ENCODER_CKPT`, or the latest matching `encoder_results/<input>/<method>/<size>/*/encoder.pth`. `as_image` sends the selected tactile representation through a visual path. `marker_only` is already three-channel RGB (`R = G = B`, white marker dots on black), shared by the `as_image` and `encode` paths. Older one-channel marker encoder checkpoints require retraining or explicit weight conversion. Set `CAMERAS=head|all|auto`; `auto` uses both cameras only when every selected task is configured for both.

The per-policy `base_config.yml` files hold model and optimizer defaults. Routing, task selection, data representation, training budget, and output destination are set in `train_policy.sh` using environment variables. PI0.5 and GR00T retain the StarVTLA optimizer defaults, including AdamW at `2.5e-5` and a warmup followed by cosine decay. Their internal horizon is 32 by default; ACT uses 50. Both VLA policies use an **absolute 8D joint** state and action, with the first action at `t+1`.

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

# Original baseline deploy configs are still accepted by the public testing entry point.
bash test_policy.sh lift_can demo ACT/deploy 0
```

`--checkpoint` accepts a physical run directory or a particular `checkpoint_<step>.pt` file; directories select the greatest available step. `CHECKPOINT=...` can be used instead of the flag. Keep `train_config.yml`, `dataset_stats.json`, and `episodes.json` beside the weights when transferring a run. `--models-root` overrides the local `resources/pretrained_models/` directory. The unified loader restores model architecture, camera routing, tactile representation, state normalization, and action denormalization from the saved training metadata. VLA tactile backbones are reconstructed from saved architecture metadata and restored from the policy state, so they do not depend on the original server's absolute encoder path. Language conditioning uses the same task-name labels used in training.

`--inspect` is a metadata check; `--offline` checks checkpoint loading and action inference. Neither reports simulator task success. Full simulator evaluation on the 5090 server remains unverified. `scripts/eval_policy.sh` is a compatibility wrapper that forwards to `test_policy.sh`; `scripts/parallel_eval.sh` retains parallel evaluation with an explicit deploy configuration.

## Distributed training and the 32-GPU experiment grid

`NUM_PROCESSES=4 bash train_policy.sh` launches one experiment with four DDP ranks using `torchrun`. Select its GPUs with `CUDA_VISIBLE_DEVICES=0,1,2,3`. `BATCH_SIZE` is the **global** batch size and must be divisible by the number of ranks: `BATCH_SIZE=64 NUM_PROCESSES=4` gives 16 samples per GPU. `WORKERS` is the number of DataLoader CPU workers **per rank**, not the GPU count. A training step is one synchronized optimizer update across all ranks. Checkpoints and validation are handled by rank 0; checkpoint model keys have no DDP `module.` prefix. Multi-task sampling retains task balancing, sharding one deterministic weighted draw across ranks and refreshing it when the loader completes an epoch.

Run the following four scripts on four separate eight-GPU machines, one script per machine. Each script launches two independent four-GPU DDP experiments on GPU groups `0,1,2,3` and `4,5,6,7`, running six methods in three rounds. Across the four machines, up to eight experiments run concurrently; the 24 configurations do not overlap.

| Machine | Script | Tactile input | Tactile type |
| --- | --- | --- | --- |
| 1 | `scripts/train_starvla_groot_node1.sh` | `marker_rgb` | `cls` |
| 2 | `scripts/train_starvla_groot_node2.sh` | `marker_rgb` | `full` |
| 3 | `scripts/train_starvla_groot_node3.sh` | `depth_deform` | `cls` |
| 4 | `scripts/train_starvla_groot_node4.sh` | `depth_deform` | `full` |

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
