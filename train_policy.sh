#!/usr/bin/env bash
# Unified UniVTAC policy training. See policy/TRAINING.md for examples.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"        # GPUs per experiment; 1 = ordinary Python
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: [environment variables] bash train_policy.sh"
  echo "NUM_PROCESSES=4 enables four-GPU DDP; BATCH_SIZE is the global batch size."
  echo "See policy/TRAINING.md for configuration examples."
  exit 0
fi
if (($#)); then echo "Unknown argument: $1" >&2; exit 2; fi
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer" >&2
  exit 2
fi

POLICY="${POLICY:-starvla_groot}"                    # act | pi05 | starvla_groot
TRAINING_MODE="${TRAINING_MODE:-multi_task}" # single_task | multi_task
TASK="${TASK:-}"                   # single_task only
TASKS="${TASKS:-}"                         # multi_task comma-separated; empty = all
EPISODES="${EPISODES:-0}"                 # per task; 0 = all
CAMERAS="${CAMERAS:-all}"                  # auto | head | all

TACTILE_MODE="${TACTILE_MODE:-encode}"     # none | as_image | encode
TACTILE_INPUT_MODE="${TACTILE_INPUT_MODE:-${INPUT_MODE:-marker_rgb}}" # marker_only | rgb_only | marker_rgb | depth_deform
TACTILE_TYPE="${TACTILE_TYPE:-full}"        # cls | full (encode only)
TACTILE_INSERT_LOCATION="${TACTILE_INSERT_LOCATION:-encoder}"
ENCODER_METHOD="${ENCODER_METHOD:-VAE}"
ENCODER_SIZE="${ENCODER_SIZE:-S}"
ENCODER_CKPT="${ENCODER_CKPT:-}"

STEPS="${STEPS:-100_0000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
WORKERS="${WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-10_000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
LOG_FREQ="${LOG_FREQ:-100}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-4}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/policy_results}"

case "$POLICY" in
  act)
    STEPS="${STEPS:-4000}"
    CHUNK_SIZE="${CHUNK_SIZE:-50}"
    if [[ -z "$BATCH_SIZE" ]]; then
      if [[ "$TACTILE_TYPE" == full ]]; then BATCH_SIZE=16; else BATCH_SIZE=64; fi
    fi
    ;;
  pi05|starvla_groot)
    STEPS="${STEPS:-20000}"
    CHUNK_SIZE="${CHUNK_SIZE:-32}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    ;;
  *) echo "Unknown POLICY: $POLICY" >&2; exit 2 ;;
esac

if [[ "$TACTILE_MODE" == encode && -z "$ENCODER_CKPT" ]]; then
  ENCODER_CKPT="$(find "$ROOT/encoder_results/$TACTILE_INPUT_MODE/$ENCODER_METHOD/$ENCODER_SIZE" -name encoder.pth -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$ENCODER_CKPT" ]]; then
    echo "No encoder checkpoint found; set ENCODER_CKPT or run train_encoder.sh" >&2
    exit 2
  fi
fi

TASK_LABEL="$TASK"
if [[ "$TRAINING_MODE" == multi_task ]]; then TASK_LABEL="${TASKS//,/_}"; TASK_LABEL="${TASK_LABEL:-all}"; fi
RUN_LABEL="${POLICY}_${TRAINING_MODE}_${TASK_LABEL}_${TACTILE_MODE}"
if [[ "$TACTILE_MODE" != none ]]; then RUN_LABEL+="_${TACTILE_INPUT_MODE}"; fi
if [[ "$TACTILE_MODE" == encode ]]; then
  RUN_LABEL+="_${ENCODER_METHOD}_${ENCODER_SIZE}_${TACTILE_TYPE}_${TACTILE_INSERT_LOCATION}"
fi
RUN_NAME="$(date +%Y%m%d-%H%M%S-%N)_${RUN_LABEL}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
mkdir -p "$(dirname "$OUTPUT_DIR")"
args=(
  --policy "$POLICY" --training-mode "$TRAINING_MODE" --task "$TASK" --tasks "$TASKS"
  --episodes "$EPISODES" --cameras "$CAMERAS"
  --tactile-mode "$TACTILE_MODE" --tactile-input-mode "$TACTILE_INPUT_MODE"
  --tactile-type "$TACTILE_TYPE" --tactile-insert-location "$TACTILE_INSERT_LOCATION"
  --steps "$STEPS" --batch-size "$BATCH_SIZE" --chunk-size "$CHUNK_SIZE"
  --workers "$WORKERS" --save-freq "$SAVE_FREQ" --log-freq "$LOG_FREQ"
  --max-val-batches "$MAX_VAL_BATCHES"
  --seed "$SEED" --device "$DEVICE" --image-size "$IMAGE_SIZE"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$ENCODER_CKPT" ]]; then args+=(--encoder-ckpt "$ENCODER_CKPT"); fi
if [[ "$SAVE_CHECKPOINT" == false ]]; then args+=(--no-save-checkpoint); fi
args+=(--encoder-method "$ENCODER_METHOD" --encoder-size "$ENCODER_SIZE")
if ((NUM_PROCESSES > 1)); then
  # --standalone selects an unused rendezvous port, allowing concurrent jobs.
  exec "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node="$NUM_PROCESSES" policy/train_unified.py "${args[@]}"
fi
exec "$PYTHON_BIN" -u policy/train_unified.py "${args[@]}"
