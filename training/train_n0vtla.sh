#!/usr/bin/env bash
# N0VTLA: bash training/train_n0vtla.sh
# 8 GPU(s), global batch 64 = 8 samples per GPU; RGB cameras, no tactile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Python environment. Set PYTHON_BIN to use another environment directly.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  source /mnt/yichi_data/miniconda3/etc/profile.d/conda.sh
  conda activate /mnt/yichi_data/miniconda3/envs/univtac
  export PYTHON_BIN="$(command -v python)"
fi

# GPU indices must be unique and match the number of DDP processes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if ((${#GPU_IDS[@]} != 8)) || [[ "$CUDA_VISIBLE_DEVICES" == *, ]]; then
  echo "CUDA_VISIBLE_DEVICES must specify exactly 8 GPU(s); got: $CUDA_VISIBLE_DEVICES" >&2
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

export NUM_PROCESSES=8

# Model, data selection and action representation.
export POLICY=n0vtla
export ACTION_MODE="${ACTION_MODE:-absolute_joint}"
export TRAINING_MODE=multi_task
export IMAGE_SIZE=224

# Training budget. BATCH_SIZE is global, shared across all GPUs.
export STEPS=50000
export BATCH_SIZE=64
export CHUNK_SIZE=50
export N_ACTION_STEPS=30
export SAVE_FREQ=10000

# Vision-only training: no tactile encoder checkpoint is required.
export TACTILE_MODE=none
export TACTILE_INPUT_MODE=rgb_only
export ENCODER_METHOD=RVAE
export ENCODER_SIZE=S

# N0 pretrained weights and PaliGemma tokenizer.
export N0_PRETRAINED_PATH="${N0_PRETRAINED_PATH:-$ROOT/resources/pretrained_models/n0-vtla-base}"
export N0_TOKENIZER_PATH="${N0_TOKENIZER_PATH:-$ROOT/resources/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer}"

echo "$POLICY: tactile=$TACTILE_MODE, action=$ACTION_MODE, training=$TRAINING_MODE, task=${TASK:-all}, tasks=${TASKS:-all}, episodes=${EPISODES:-0}, cameras=${CAMERAS:-all}, image_size=$IMAGE_SIZE, GPUs=$CUDA_VISIBLE_DEVICES, global_batch=$BATCH_SIZE, steps=$STEPS, save_freq=$SAVE_FREQ, chunk_size=$CHUNK_SIZE, n_action_steps=$N_ACTION_STEPS"
exec bash "$ROOT/train_policy.sh" "$@"
