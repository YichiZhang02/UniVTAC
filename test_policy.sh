#!/usr/bin/env bash
# Usage: bash test_policy.sh [MODEL_ID [STEP [GPU_IDS [NUM_GPU]]]] [--dry-run]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ $# -eq 1 && "$1" == "--dry-run" ]]; then
  set -- "" "" "" "" "--dry-run"
fi

MODEL_ID=${1:-20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder}
STEP=${2:-30000}
CUDA_VISIBLE_DEVICES=${3:-0,1,2,3,4,5,6,7}
NUM_GPU=${4:-8}
TASK_CONFIG="demo"
TOTAL_NUM=20                           # Valid evaluation episodes per task.
START_SEED=-1                          # -1 resolves to 1000000 for this model.
MAX_SEED=-1                            # No seed ceiling.
MAX_ERRORS=10                          # Stop a task after repeated exceptions.
RESET_TIME_LIMIT=600                   # This host exceeded the default 120 s reset limit.

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT/../miniconda3/envs/univtac/bin/python" ]]; then
    PYTHON_BIN="$ROOT/../miniconda3/envs/univtac/bin/python"
  else
    PYTHON_BIN=python
  fi
fi

# Preserve named CLI and legacy forms. The positional form below has defaults.
if [[ "${1:-}" == --* && "${1:-}" != "--dry-run" ]]; then
  for arg in "$@"; do
    if [[ "$arg" == "--model-id" || "$arg" == --model-id=* ]]; then
      exec "$PYTHON_BIN" -u "$ROOT/scripts/test_policy.py" \
        --task-config "$TASK_CONFIG" --total-num "$TOTAL_NUM" \
        --start-seed "$START_SEED" --max-seed "$MAX_SEED" --max-errors "$MAX_ERRORS" \
        --config-overrides "task_overrides.reset_time_limit=$RESET_TIME_LIMIT" "$@"
    fi
  done
  exec "$PYTHON_BIN" -u "$ROOT/scripts/test_policy.py" "$@"
fi

if (( $# > 4 )); then
  extra_args=("${@:5}")
else
  extra_args=()
fi



exec "$PYTHON_BIN" -u "$ROOT/scripts/test_policy.py" \
  --task-config "$TASK_CONFIG" --total-num "$TOTAL_NUM" \
  --start-seed "$START_SEED" --max-seed "$MAX_SEED" --max-errors "$MAX_ERRORS" \
  --config-overrides "task_overrides.reset_time_limit=$RESET_TIME_LIMIT" \
  --model-id "$MODEL_ID" --step "$STEP" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" --num-gpu "$NUM_GPU" \
  "${extra_args[@]}"
