#!/usr/bin/env bash
# One eight-GPU node: two four-GPU DDP jobs at a time, six methods in three waves.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:?Missing tactile input mode}"
TYPE="${2:?Missing tactile type}"
shift 2
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash scripts/train_starvla_groot_node<N>.sh [--dry-run]"
  echo "GPUS=0,1,2,3,4,5,6,7; two four-GPU DDP experiments concurrently."
  echo "Global batch=64; per-GPU batch=16; steps=30000; save_freq=5000."
  echo "Override PYTHON_BIN, GPUS, OUTPUT_ROOT, WORKERS, or RUN_ID using environment variables."
  echo "Each result directory contains train.log, launcher.log and exit_code."
  exit 0
fi
DRY_RUN=0
if [[ "${1:-}" == --dry-run && $# == 1 ]]; then
  DRY_RUN=1
elif (($#)); then
  echo "Unknown arguments: $*" >&2
  exit 2
fi
case "$INPUT/$TYPE" in marker_rgb/cls|marker_rgb/full|depth_deform/cls|depth_deform/full) ;; *) exit 2 ;; esac
if ! PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"; then
  echo "Python not found; set PYTHON_BIN for this machine." >&2
  exit 2
fi
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%N)}"
NODE_LABEL="${INPUT}_${TYPE}"
OUT_ROOT="${OUTPUT_ROOT:-$ROOT/policy_results/starvla_groot_${NODE_LABEL}_${RUN_ID}}"
WORKERS="${WORKERS:-4}" # Per DDP rank: 8 processes x 4 workers on this node.
METHODS=(ResNet VAE MAE DINOv2 I-JEPA V-JEPA)
declare -A CKPT
CKPT[marker_rgb/ResNet]="$ROOT/encoder_results/marker_rgb/ResNet/S/20260930-200744-984819/encoder.pth"
CKPT[marker_rgb/VAE]="$ROOT/encoder_results/marker_rgb/VAE/S/20260930-204950-862027/encoder.pth"
CKPT[marker_rgb/MAE]="$ROOT/encoder_results/marker_rgb/MAE/S/20260930-211901-297263/encoder.pth"
CKPT[marker_rgb/DINOv2]="$ROOT/encoder_results/marker_rgb/DINOv2/S/20260930-214410-965503/encoder.pth"
CKPT[marker_rgb/I-JEPA]="$ROOT/encoder_results/marker_rgb/I-JEPA/S/20261001-073312-844529/encoder.pth"
CKPT[marker_rgb/V-JEPA]="$ROOT/encoder_results/marker_rgb/V-JEPA/S/20261001-073312-845111/encoder.pth"
CKPT[depth_deform/ResNet]="$ROOT/encoder_results/depth_deform/ResNet/S/20261001-073241-879741/encoder.pth"
CKPT[depth_deform/VAE]="$ROOT/encoder_results/depth_deform/VAE/S/20261001-073241-880252/encoder.pth"
CKPT[depth_deform/MAE]="$ROOT/encoder_results/depth_deform/MAE/S/20261001-073241-880974/encoder.pth"
CKPT[depth_deform/DINOv2]="$ROOT/encoder_results/depth_deform/DINOv2/S/20261001-073241-881520/encoder.pth"
CKPT[depth_deform/I-JEPA]="$ROOT/encoder_results/depth_deform/I-JEPA/S/20261001-080316-675871/encoder.pth"
CKPT[depth_deform/V-JEPA]="$ROOT/encoder_results/depth_deform/V-JEPA/S/20261001-085004-403968/encoder.pth"
IFS=',' read -r -a GPU_IDS <<< "$GPUS_CSV"
if ((${#GPU_IDS[@]} != 8)); then
  echo "GPUS must contain eight distinct GPU indices; got $GPUS_CSV" >&2
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
GPU_GROUPS=("${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]},${GPU_IDS[3]}"
            "${GPU_IDS[4]},${GPU_IDS[5]},${GPU_IDS[6]},${GPU_IDS[7]}")
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN; set PYTHON_BIN for this machine." >&2
  exit 2
fi
for method in "${METHODS[@]}"; do
  if [[ ! -s "${CKPT[$INPUT/$method]}" ]]; then
    echo "Missing encoder checkpoint: ${CKPT[$INPUT/$method]}" >&2
    exit 2
  fi
done
if ((DRY_RUN)); then
  for index in "${!METHODS[@]}"; do
    method="${METHODS[$index]}"
    slot=$((index % 2))
    echo "wave=$((index / 2 + 1)) GPUs=${GPU_GROUPS[$slot]} input=$INPUT type=$TYPE method=$method size=S location=encoder global_batch=64 per_gpu_batch=16 steps=30000 save_freq=5000 checkpoint=${CKPT[$INPUT/$method]}"
  done
  exit 0
fi
# Check visibility before starting any expensive model loading.
CUDA_VISIBLE_DEVICES="$GPUS_CSV" "$PYTHON_BIN" -c \
  'import torch; n=torch.cuda.device_count(); assert n == 8, f"Expected eight visible GPUs, got {n}"'
mkdir -p "$OUT_ROOT"
run_worker() {
  local slot="$1" index method name log out rc failures=0 job_pid=""
  trap 'if [[ -n "$job_pid" ]]; then kill -TERM "$job_pid" 2>/dev/null || true; wait "$job_pid" 2>/dev/null || true; fi; exit 130' INT TERM
  for ((index=slot; index<${#METHODS[@]}; index+=2)); do
    method="${METHODS[$index]}"
    name="${INPUT}_${method}_S_${TYPE}_encoder"
    out="$OUT_ROOT/${RUN_ID}_starvla_groot_multi_task_all_encode_${INPUT}_${method}_S_${TYPE}_encoder"
    # Reserve each output before redirecting the launcher; never reuse a run.
    if ! mkdir "$out"; then
      echo "Result directory already exists or cannot be created: $out" >&2
      return 1
    fi
    log="$out/launcher.log"
    echo "[$(date -Is)] START $name GPUs=${GPU_GROUPS[$slot]} log=$log"
    env PYTHON_BIN="$PYTHON_BIN" CUDA_VISIBLE_DEVICES="${GPU_GROUPS[$slot]}" NUM_PROCESSES=4 \
      POLICY=starvla_groot TRAINING_MODE=multi_task TASKS='' EPISODES=0 CAMERAS=all \
      TACTILE_MODE=encode TACTILE_INPUT_MODE="$INPUT" TACTILE_TYPE="$TYPE" \
      TACTILE_INSERT_LOCATION=encoder ENCODER_METHOD="$method" ENCODER_SIZE=S \
      ENCODER_CKPT="${CKPT[$INPUT/$method]}" STEPS=30000 BATCH_SIZE=64 CHUNK_SIZE=32 \
      SAVE_FREQ=5000 SAVE_CHECKPOINT=true WORKERS="$WORKERS" DEVICE=cuda OUTPUT_DIR="$out" \
      bash "$ROOT/train_policy.sh" >"$log" 2>&1 &
    job_pid=$!
    if wait "$job_pid"; then
      rc=0
      echo "[$(date -Is)] DONE $name"
    else
      rc=$?
      failures=$((failures + 1))
      echo "[$(date -Is)] FAIL($rc) $name log=$log" >&2
    fi
    job_pid=""
    echo "$rc" >"$out/exit_code"
  done
  return "$failures"
}
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM
echo "Node $NODE_LABEL: six experiments; two concurrent four-GPU DDP jobs."
echo "Output: $OUT_ROOT"
echo "Logs: each result directory (train.log, launcher.log)"
for slot in 0 1; do
  run_worker "$slot" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else failed=1; fi
done
if ((failed)); then
  echo "Some experiments failed. See each result's exit_code and launcher.log under $OUT_ROOT" >&2
  exit 1
fi
echo "All six experiments completed."
