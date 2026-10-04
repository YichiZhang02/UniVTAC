#!/usr/bin/env bash
# Compatibility helper; the public testing entry point is test_policy.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/test_policy.sh" "$@"
