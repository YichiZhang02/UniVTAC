#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/mnt/yichi_data/miniconda3/envs/univtac/bin/python}"
GPUS="${GPUS:-4}"
export PYTHONPATH="$PWD:$PWD/policy/starvla_groot/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" -m training.train "$@"
