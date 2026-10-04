#!/usr/bin/env bash
# Train one tactile encoder. Optimization defaults live in encoder/<method>/config.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
METHOD="${ENCODER_METHOD:-ResNet}"
SIZE="${ENCODER_SIZE:-${VIT_SIZE:-S}}"
INPUT="${INPUT_MODE:-marker_rgb}"
DATA_ROOT="$ROOT/resources/data/contact"
CONFIG=""
EXTRA=()
usage() {
    echo "Usage: $0 [--method ResNet|VAE|MAE|DINOv2|I-JEPA|V-JEPA] [--size S|B] [--input marker_only|rgb_only|marker_rgb|depth_deform] [--data-root DIR] [--config FILE] [-- additional train.py options]"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
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
case "$METHOD" in ResNet|VAE|MAE|DINOv2|I-JEPA|V-JEPA) ;; *) echo "Invalid model: $METHOD" >&2; exit 2 ;; esac
case "$SIZE" in S|B) ;; *) echo "Invalid size: $SIZE" >&2; exit 2 ;; esac
case "$INPUT" in marker_only|rgb_only|marker_rgb|depth_deform) ;; *) echo "Invalid input: $INPUT" >&2; exit 2 ;; esac
CONFIG="${CONFIG:-$ROOT/encoder/$METHOD/config.yml}"
RUN_DIR="$ROOT/encoder_results/$INPUT/$METHOD/$SIZE/$(date +%Y%m%d-%H%M%S-%6N)"
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
    echo "Model=$METHOD size=$SIZE input=$INPUT config=$CONFIG output=$RUN_DIR"
    "$PYTHON_BIN" -u "$ROOT/encoder/$METHOD/train.py" --size "$SIZE" --input-mode "$INPUT" \
        --data-root "$DATA_ROOT" --config "$CONFIG" --output-dir "$RUN_DIR" "${EXTRA[@]}"
} 2>&1 | tee "${TEE_ARGS[@]}" "$LOG"
