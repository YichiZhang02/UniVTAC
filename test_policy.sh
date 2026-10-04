#!/usr/bin/env bash
# Test train_policy.sh checkpoints, or launch a legacy deploy configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u "$ROOT/scripts/test_policy.py" "$@"
