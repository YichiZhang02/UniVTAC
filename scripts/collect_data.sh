#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${1:-}" == --help || "${1:-}" == -h || $# -lt 2 ]]; then
  echo "Usage: bash scripts/collect_data.sh TASK CONFIG [GPU] [START_SEED] [MAX_SEED] [EPISODES]"
  exit 0
fi
TASK_NAME="$1"
CONFIG_NAME="$2"
GPU="${3:-0}"
START_SEED="${4:--1}"
MAX_SEED="${5:--1}"
EPISODE="${6:--1}"
CMD=("$PYTHON_BIN" scripts/collect_data.py "$TASK_NAME" "$CONFIG_NAME" --start_seed "$START_SEED" --max_seed "$MAX_SEED" --gpu "$GPU")
if [[ "$EPISODE" != -1 ]]; then CMD+=(--config-overrides "collect_settings.episode_num=$EPISODE"); fi
exec "${CMD[@]}"
