#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${1:-}" == --help || "${1:-}" == -h || $# -lt 1 ]]; then
  echo "Usage: bash scripts/parallel_collect.sh TASK [CONFIG] [GPU] [WORKERS]"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="${3:-0}"
exec "$PYTHON_BIN" scripts/parallel_collect_data.py "$1" "${2:-demo}" --workers="${4:-3}"
