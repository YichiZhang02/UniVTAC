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
  echo "CHUNK_SIZE=50 predicts 50 actions; N_ACTION_STEPS=30 executes 30 before replanning."
  echo "See policy/TRAINING.md for configuration examples."
  exit 0
fi
if (($#)); then echo "Unknown argument: $1" >&2; exit 2; fi
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer" >&2
  exit 2
fi

POLICY="${POLICY:-starvla_groot}"                    # act | pi05 | starvla_groot | starvla_pi | n0vtla
if [[ "$POLICY" == n0vtla ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  ACTION_MODE="${ACTION_MODE:-relative_joint}"
  TACTILE_MODE="${TACTILE_MODE:-native}"
  TACTILE_INPUT_MODE="${TACTILE_INPUT_MODE:-rgb_only}"
  CHUNK_SIZE="${CHUNK_SIZE:-50}"
  STEPS="${STEPS:-20000}"
fi
TRAINING_MODE="${TRAINING_MODE:-multi_task}" # single_task | multi_task
TASK="${TASK:-}"                   # single_task only
TASKS="${TASKS:-}"                         # multi_task comma-separated; empty = all
EPISODES="${EPISODES:-0}"                 # per task; 0 = all
CAMERAS="${CAMERAS:-all}"                  # auto | head | all
ACTION_MODE="${ACTION_MODE:-relative_joint}" # absolute[_joint] | relative[_joint]; gripper stays absolute
case "$ACTION_MODE" in
  absolute|absolute_joint) ACTION_MODE=absolute_joint ;;
  relative|relative_joint) ACTION_MODE=relative_joint ;;
  *) echo "Unknown ACTION_MODE: $ACTION_MODE" >&2; exit 2 ;;
esac

TACTILE_MODE="${TACTILE_MODE:-encode}"     # none | as_image | encode | native (N0)
TACTILE_INPUT_MODE="${TACTILE_INPUT_MODE:-${INPUT_MODE:-marker_rgb}}" # marker_only | rgb_only | marker_rgb | depth_deform_20 | depth_deform_80 | depth_deform_160 | depth_deform_240 | depth_deform_nonlinear | depth_deform_nonlinear_radial
TACTILE_TYPE="${TACTILE_TYPE:-full}"        # cls | full | pool3x3 | pool5x5 | pool7x7 (pool aliases pool3x3; encode only)
TACTILE_INSERT_LOCATION="${TACTILE_INSERT_LOCATION:-encoder}"
ENCODER_PRETRAINED_MODE="${ENCODER_PRETRAINED_MODE:-univtac}" # univtac | tactile | vision
ENCODER_METHOD="${ENCODER_METHOD:-VAE}"
ENCODER_SIZE="${ENCODER_SIZE:-}"
ENCODER_CKPT="${ENCODER_CKPT:-}"
ENCODER_INIT="${ENCODER_INIT:-latest}" # latest | scratch (alias) | official (raw official weights)
if [[ "$ENCODER_INIT" == scratch ]]; then
  ENCODER_PRETRAIN_INIT=scratch
  ENCODER_INIT=latest
fi
if [[ "$ENCODER_PRETRAINED_MODE" == univtac ]]; then
  ENCODER_PRETRAIN_INIT="${ENCODER_PRETRAIN_INIT:-scratch}"
else
  ENCODER_PRETRAIN_INIT="${ENCODER_PRETRAIN_INIT:-official}"
fi
case "$ENCODER_PRETRAIN_INIT" in scratch|official) ;; *) echo "ENCODER_PRETRAIN_INIT must be scratch or official" >&2; exit 2 ;; esac
case "$ENCODER_INIT" in
  latest) ;;
  official)
    if [[ "$ENCODER_PRETRAIN_INIT" != official ]]; then
      echo "ENCODER_INIT=official requires ENCODER_PRETRAIN_INIT=official" >&2; exit 2
    fi
    if [[ "$TACTILE_MODE" != encode || -n "$ENCODER_CKPT" ]] || \
       [[ "$ENCODER_PRETRAINED_MODE" != tactile && ! ( "$ENCODER_PRETRAINED_MODE" == vision && "$ENCODER_METHOD" == Wan2.2-VAE ) ]]; then
      echo "ENCODER_INIT=official requires tactile or vision/Wan2.2-VAE encode mode and no ENCODER_CKPT" >&2
      exit 2
    fi
    ;;
  *) echo "Unknown ENCODER_INIT: $ENCODER_INIT" >&2; exit 2 ;;
esac

STEPS="${STEPS:-5_0000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-30}" # Actions executed before replanning
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
  pi05|starvla_groot|starvla_pi|n0vtla)
    STEPS="${STEPS:-20000}"
    CHUNK_SIZE="${CHUNK_SIZE:-50}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    ;;
  *) echo "Unknown POLICY: $POLICY" >&2; exit 2 ;;
esac

if [[ ! "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ || ! "$N_ACTION_STEPS" =~ ^[1-9][0-9]*$ ]] || ((N_ACTION_STEPS > CHUNK_SIZE)); then
  echo "Require positive integers with 1 <= N_ACTION_STEPS <= CHUNK_SIZE; got $N_ACTION_STEPS and $CHUNK_SIZE" >&2
  exit 2
fi

if [[ -z "$ENCODER_SIZE" ]]; then
  ENCODER_SIZE="$("$PYTHON_BIN" -c 'import sys; from encoder.pretrained_registry import default_size; print(default_size(*sys.argv[1:]))' "$ENCODER_PRETRAINED_MODE" "$ENCODER_METHOD")"
fi
if [[ "$TACTILE_MODE" == encode && -z "$ENCODER_CKPT" && "$ENCODER_INIT" != official ]]; then
  ENCODER_CKPT="$("$PYTHON_BIN" -c 'import sys; from encoder.pretrained_registry import latest_checkpoint; print(latest_checkpoint(*sys.argv[1:]) or "")' "$TACTILE_INPUT_MODE" "$ENCODER_PRETRAINED_MODE" "$ENCODER_METHOD" "$ENCODER_SIZE" "$ENCODER_PRETRAIN_INIT")"
  if [[ -z "$ENCODER_CKPT" && ( "$ENCODER_PRETRAINED_MODE" != tactile || "$ENCODER_PRETRAIN_INIT" == scratch ) ]]; then
    echo "No $ENCODER_PRETRAIN_INIT encoder checkpoint found; set ENCODER_CKPT or run train_encoder.sh --init $ENCODER_PRETRAIN_INIT" >&2
    exit 2
  fi
fi

TASK_LABEL="$TASK"
if [[ "$TRAINING_MODE" == multi_task ]]; then TASK_LABEL="${TASKS//,/_}"; TASK_LABEL="${TASK_LABEL:-all}"; fi
RUN_LABEL="${POLICY}_${TRAINING_MODE}_${TASK_LABEL}_${TACTILE_MODE}"
if [[ "$ACTION_MODE" == relative_joint ]]; then RUN_LABEL+="_relative_joint"; fi
if [[ "$TACTILE_MODE" != none ]]; then RUN_LABEL+="_${TACTILE_INPUT_MODE}"; fi
if [[ "$TACTILE_MODE" == encode ]]; then
  RUN_LABEL+="_${ENCODER_PRETRAINED_MODE}_${ENCODER_METHOD}_${ENCODER_SIZE}_${ENCODER_PRETRAIN_INIT}_${TACTILE_TYPE}_${TACTILE_INSERT_LOCATION}"
fi
if [[ -n "${RUN_TAG:-}" ]]; then RUN_LABEL+="_${RUN_TAG}"; fi
RUN_NAME="$(date +%Y%m%d-%H%M%S-%N)_${RUN_LABEL}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
mkdir -p "$(dirname "$OUTPUT_DIR")"
args=(
  --policy "$POLICY" --training-mode "$TRAINING_MODE" --task "$TASK" --tasks "$TASKS"
  --episodes "$EPISODES" --cameras "$CAMERAS"
  --action-mode "$ACTION_MODE"
  --tactile-mode "$TACTILE_MODE" --tactile-input-mode "$TACTILE_INPUT_MODE"
  --tactile-type "$TACTILE_TYPE" --tactile-insert-location "$TACTILE_INSERT_LOCATION"
  --tactile-preprocessing "${TACTILE_PREPROCESSING:-auto}"
  --wan-training-optimization "${WAN_TRAINING_OPTIMIZATION:-reference}"
  --steps "$STEPS" --batch-size "$BATCH_SIZE" --chunk-size "$CHUNK_SIZE" --n-action-steps "$N_ACTION_STEPS"
  --workers "$WORKERS" --save-freq "$SAVE_FREQ" --log-freq "$LOG_FREQ"
  --max-val-batches "$MAX_VAL_BATCHES"
  --seed "$SEED" --device "$DEVICE" --image-size "$IMAGE_SIZE"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$ENCODER_CKPT" ]]; then args+=(--encoder-ckpt "$ENCODER_CKPT"); fi
if [[ "$POLICY" == n0vtla ]]; then
  args+=(--pretrained-path "${N0_PRETRAINED_PATH:-$ROOT/resources/pretrained_models/n0-vtla-base}"
         --tokenizer-path "${N0_TOKENIZER_PATH:-$ROOT/resources/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer}")
fi
if [[ "$SAVE_CHECKPOINT" == false ]]; then args+=(--no-save-checkpoint); fi
args+=(--encoder-pretrained-mode "$ENCODER_PRETRAINED_MODE" --encoder-method "$ENCODER_METHOD" --encoder-size "$ENCODER_SIZE")
args+=(--encoder-pretrain-init "$ENCODER_PRETRAIN_INIT")
if ((NUM_PROCESSES > 1)); then
  # --standalone selects an unused rendezvous port, allowing concurrent jobs.
  exec "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node="$NUM_PROCESSES" policy/train_unified.py "${args[@]}"
fi
exec "$PYTHON_BIN" -u policy/train_unified.py "${args[@]}"
