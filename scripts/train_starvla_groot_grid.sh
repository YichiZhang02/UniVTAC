#!/usr/bin/env bash
# Run 24 StarVLA-GR00T tactile ablations across four GPUs.
# Each run uses one GPU; four independent runs execute concurrently (not DDP).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"; then
  echo "Python not found; set PYTHON_BIN for this machine." >&2
  exit 2
fi
GPUS_CSV="${GPUS:-0,1,2,3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUTPUT_ROOT:-$ROOT/policy_results/starvla_groot_grid_$RUN_ID}"
WORKER_LOG_ROOT="$OUT_ROOT/logs"
WORKERS="${WORKERS:-4}"

METHODS=(ResNet VAE MAE DINOv2 I-JEPA V-JEPA)
INPUTS=(marker_rgb depth_deform)
TYPES=(cls full)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/train_starvla_groot_grid.sh [--dry-run]"
  echo "Runs 24 configurations, four at a time on GPUs listed by GPUS (default 0,1,2,3)."
  echo "This schedules independent single-GPU runs; it does not use DDP."
  exit 0
fi

# Fixed, previously trained S checkpoints, grouped by tactile input mode.
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

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if ((${#GPUS[@]} != 4)); then
  echo "Expected four GPU IDs in GPUS (default: 0,1,2,3); got: $GPUS_CSV" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN (override with PYTHON_BIN=...)" >&2
  exit 2
fi
for input in "${INPUTS[@]}"; do
  for method in "${METHODS[@]}"; do
    ckpt="${CKPT[$input/$method]}"
    if [[ ! -s "$ckpt" ]]; then
      echo "Missing trained encoder checkpoint: $ckpt" >&2
      exit 2
    fi
  done
done

if [[ "${1:-}" == "--dry-run" ]]; then
  for input in "${INPUTS[@]}"; do
    for method in "${METHODS[@]}"; do
      for type in "${TYPES[@]}"; do
        echo "$input $method S $type encoder checkpoint=${CKPT[$input/$method]}"
      done
    done
  done
  exit 0
elif [[ -n "${1:-}" ]]; then
  echo "Unknown option: $1" >&2
  exit 2
fi

mkdir -p "$WORKER_LOG_ROOT"
run_worker() {
  local gpu="$1" slot="$2" input method type ckpt run_name out log index=0 failures=0 rc
  for input in "${INPUTS[@]}"; do
    for method in "${METHODS[@]}"; do
      for type in "${TYPES[@]}"; do
        if ((index % ${#GPUS[@]} == slot)); then
          ckpt="${CKPT[$input/$method]}"
          run_name="${input}_${method}_S_${type}_encoder"
          out="$OUT_ROOT/${RUN_ID}_starvla_groot_multi_task_all_encode_${input}_${method}_S_${type}_encoder"
          if ! mkdir "$out"; then
            echo "Result directory already exists or cannot be created: $out" >&2
            return 1
          fi
          log="$out/launcher.log"
          echo "[$(date -Is)] GPU=$gpu START $run_name checkpoint=$ckpt"
          if env \
            PYTHON_BIN="$PYTHON_BIN" CUDA_VISIBLE_DEVICES="$gpu" \
            POLICY=starvla_groot TRAINING_MODE=multi_task TASKS='' EPISODES=0 CAMERAS=all \
            TACTILE_MODE=encode TACTILE_INPUT_MODE="$input" TACTILE_TYPE="$type" \
            TACTILE_INSERT_LOCATION=encoder ENCODER_METHOD="$method" ENCODER_SIZE=S \
            ENCODER_CKPT="$ckpt" STEPS=30000 BATCH_SIZE=16 CHUNK_SIZE=32 \
            SAVE_FREQ=5000 WORKERS="$WORKERS" DEVICE=cuda OUTPUT_DIR="$out" \
            bash "$ROOT/train_policy.sh" >"$log" 2>&1; then
            echo 0 >"$out/exit_code"
            echo "[$(date -Is)] GPU=$gpu DONE  $run_name"
          else
            rc=$?
            echo "$rc" >"$out/exit_code"
            echo "[$(date -Is)] GPU=$gpu FAIL($rc) $run_name log=$log" >&2
            failures=$((failures + 1))
          fi
        fi
        index=$((index + 1))
      done
    done
  done
  return "$failures"
}

echo "Launching 24 runs; four concurrent single-GPU workers on GPUs: $GPUS_CSV"
echo "Output: $OUT_ROOT"
echo "Logs: each result directory; worker scheduling logs: $WORKER_LOG_ROOT"
pids=()
for slot in "${!GPUS[@]}"; do
  run_worker "${GPUS[$slot]}" "$slot" >"$WORKER_LOG_ROOT/worker_gpu${GPUS[$slot]}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else failed=1; fi
done
if ((failed)); then
  echo "One or more runs failed. Inspect result logs under $OUT_ROOT" >&2
  exit 1
fi
echo "All 24 runs completed successfully."
