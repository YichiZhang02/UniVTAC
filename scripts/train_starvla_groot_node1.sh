#!/usr/bin/env bash
# Node 1: marker_rgb, cls; six methods, two four-GPU DDP jobs concurrently.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/train_starvla_groot_node.sh" marker_rgb cls "$@"
