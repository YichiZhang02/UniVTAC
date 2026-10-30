#!/usr/bin/env bash
# StarVLA-PI: bash training/train_starvla_pi.sh
# 4 GPU(s), global batch 64 = 16 samples per GPU; RGB cameras, no tactile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Fixed training environment.
source /mnt/yichi_data/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/yichi_data/miniconda3/envs/univtac
export PYTHON_BIN="$(command -v python)"

# GPU indices must be unique and match the number of DDP processes.
export CUDA_VISIBLE_DEVICES=0,1,2,3
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if ((${#GPU_IDS[@]} != 4)) || [[ "$CUDA_VISIBLE_DEVICES" == *, ]]; then
  echo "CUDA_VISIBLE_DEVICES must specify exactly 4 GPU(s); got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
declare -A SEEN_GPU=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN_GPU[$gpu]:-}" ]]; then
    echo "Invalid or repeated GPU index: $gpu" >&2
    exit 2
  fi
  SEEN_GPU[$gpu]=1
done

export NUM_PROCESSES=4

# Model, data selection and action representation.
export POLICY=starvla_pi
export ACTION_MODE="${ACTION_MODE:-absolute_joint}"
export TRAINING_MODE=multi_task
export IMAGE_SIZE=224

# Training budget. BATCH_SIZE is global, shared across all GPUs.
export STEPS=50000
export BATCH_SIZE=64
export CHUNK_SIZE=50
export N_ACTION_STEPS=30
export SAVE_FREQ=10000

# Vision-only training; no tactile encoder checkpoint is required.
export TACTILE_MODE=none
export TACTILE_INPUT_MODE=rgb_only
export ENCODER_METHOD=RVAE
export ENCODER_SIZE=S

echo "$POLICY: tactile=$TACTILE_MODE, action=$ACTION_MODE, training=$TRAINING_MODE, task=${TASK:-all}, tasks=${TASKS:-all}, episodes=${EPISODES:-0}, cameras=${CAMERAS:-all}, image_size=$IMAGE_SIZE, GPUs=$CUDA_VISIBLE_DEVICES, global_batch=$BATCH_SIZE, steps=$STEPS, save_freq=$SAVE_FREQ, chunk_size=$CHUNK_SIZE, n_action_steps=$N_ACTION_STEPS"
exec bash "$ROOT/train_policy.sh" "$@"
