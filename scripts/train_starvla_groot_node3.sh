#!/usr/bin/env bash
# Node 3: depth_deform, cls; six methods, two four-GPU DDP jobs concurrently.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/train_starvla_groot_node.sh" depth_deform cls "$@"
