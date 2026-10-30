#!/usr/bin/env bash
# Train one tactile encoder. Optimization defaults live in encoder/<method>/config.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
PRETRAINED_MODE="${PRETRAINED_MODE:-univtac}" # univtac | tactile | vision
METHOD="${ENCODER_METHOD:-ResNet}"
SIZE="${ENCODER_SIZE:-${VIT_SIZE:-}}"
INPUT="${INPUT_MODE:-marker_rgb}"
DATA_ROOT="$ROOT/resources/data/contact"
CONFIG=""
EXTRA=()
INIT=""
PREPROCESSING="${TACTILE_PREPROCESSING:-auto}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
usage() {
    echo "Usage: $0 [--init scratch|official|adapted] [--pretrained-mode univtac|tactile|vision] [--method METHOD] [--size S|B|official] [--input INPUT_MODE] [--data-root DIR] [--config FILE] [-- additional train.py options]"
    echo "scratch S/B: ResNet RVAE RAE SD-VAE MAE DINOv2 I-JEPA V-JEPA; fixed official size: Sparsh_mae Sparsh_DINOv2 Sparsh_IJEPA Sparsh_VJEPA Wan22_VAE Anytouch2"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tactile-preprocessing) PREPROCESSING="$2"; shift 2 ;;
        --init) INIT="$2"; shift 2 ;;
        --pretrained-mode) PRETRAINED_MODE="$2"; shift 2 ;;
        --method|--model) METHOD="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --input|--input-mode) INPUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --) shift; EXTRA+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
IDENTITY="$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c 'import sys; from encoder.catalog import selection; print(*selection(*[v or None for v in sys.argv[1:]]))' "$PRETRAINED_MODE" "$METHOD" "$SIZE" "$INIT")"
read -r PRETRAINED_MODE METHOD SIZE INIT <<< "$IDENTITY"
case "$PREPROCESSING" in auto|cpu|cuda) ;; *) echo "Invalid tactile preprocessing: $PREPROCESSING" >&2; exit 2 ;; esac
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer" >&2; exit 2
fi
if [[ "$PRETRAINED_MODE" == univtac && "$INIT" == official ]]; then
    echo "univtac methods have no official initializer; use --init scratch" >&2; exit 2
fi
if [[ "$PRETRAINED_MODE" == univtac && "$NUM_PROCESSES" != 1 ]]; then
    echo "NUM_PROCESSES is supported for external encoders; local encoders use --multi-gpu" >&2; exit 2
fi
case "$INPUT" in marker_only|rgb_only|marker_rgb|depth_deform_20|depth_deform_80|depth_deform_160|depth_deform_240|depth_deform_nonlinear|depth_deform_nonlinear_radial) ;; *) echo "Invalid input: $INPUT" >&2; exit 2 ;; esac
CONFIG="${CONFIG:-$ROOT/encoder/$METHOD/config.yml}"
RUN_DIR="$ROOT/encoder_results/$INPUT/$PRETRAINED_MODE/$METHOD/$SIZE/$INIT/$(date +%Y%m%d-%H%M%S-%6N)"
for ((i=0; i<${#EXTRA[@]}; i++)); do
    if [[ "${EXTRA[i]}" == --resume && $((i + 1)) -lt ${#EXTRA[@]} ]]; then
        RUN_DIR="$(dirname "${EXTRA[i+1]}")"
    elif [[ "${EXTRA[i]}" == --resume=* ]]; then
        RUN_DIR="$(dirname "${EXTRA[i]#--resume=}")"
    fi
done
for ((i=0; i<${#EXTRA[@]}; i++)); do
    if [[ "${EXTRA[i]}" == --output-dir && $((i + 1)) -lt ${#EXTRA[@]} ]]; then
        RUN_DIR="${EXTRA[i+1]}"
    elif [[ "${EXTRA[i]}" == --output-dir=* ]]; then
        RUN_DIR="${EXTRA[i]#--output-dir=}"
    fi
done
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train.log"
TEE_ARGS=()
for arg in "${EXTRA[@]}"; do
    if [[ "$arg" == --resume || "$arg" == --resume=* ]]; then TEE_ARGS=(-a); fi
done
{
    DISPLAY_METHOD="$METHOD"
    if [[ "$PRETRAINED_MODE" == univtac && "$METHOD" == V-JEPA ]]; then DISPLAY_METHOD='V-JEPA*'; fi
    echo "Model=$DISPLAY_METHOD size=$SIZE input=$INPUT batch_size=64 config=$CONFIG output=$RUN_DIR"
    if [[ "$PRETRAINED_MODE" == univtac ]]; then
        "$PYTHON_BIN" -u "$ROOT/encoder/$METHOD/train.py" --init "$INIT" --size "$SIZE" --input-mode "$INPUT" \
            --data-root "$DATA_ROOT" --config "$CONFIG" --output-dir "$RUN_DIR" --tactile-preprocessing "$PREPROCESSING" "${EXTRA[@]}"
    else
        CONFIG_ARGS=()
        if [[ -n "$CONFIG" ]]; then CONFIG_ARGS=(--config "$CONFIG"); fi
        if [[ -n "$INIT" ]]; then CONFIG_ARGS+=(--init "$INIT"); fi
        LAUNCH=("$PYTHON_BIN" -u)
        if ((NUM_PROCESSES > 1)); then
            LAUNCH+=(-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NUM_PROCESSES")
            CONFIG_ARGS+=(--no-multi-gpu)
        fi
        "${LAUNCH[@]}" "$ROOT/encoder/train_external.py" --pretrained-mode "$PRETRAINED_MODE" --method "$METHOD" \
            --size "$SIZE" --input-mode "$INPUT" --data-root "$DATA_ROOT" --output-dir "$RUN_DIR" --tactile-preprocessing "$PREPROCESSING" "${CONFIG_ARGS[@]}" "${EXTRA[@]}"
    fi
} 2>&1 | tee "${TEE_ARGS[@]}" "$LOG"
