#!/usr/bin/env bash
# Parallel simulator evaluation with an explicit deploy configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${1:-}" == --help || "${1:-}" == -h || $# -lt 3 ]]; then
  echo "Usage: bash scripts/parallel_eval.sh TASK TASK_CONFIG DEPLOY_CONFIG [GPU] [WORKERS] [TOTAL_NUM]"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="${4:-0}"
exec "$PYTHON_BIN" scripts/parallel_eval_policy.py "$1" "$2" "$3" --total_num "${6:-100}" --workers "${5:-2}"
