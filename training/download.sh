#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PYTHON="${PYTHON:-/mnt/yichi_data/miniconda3/envs/univtac/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/mnt/yichi_data/UniVTAC_downloads}"
export TQDM_DISABLE=1
"$(dirname "$PYTHON")/modelscope" download --dataset byml2024/UniVTAC --local_dir "$DOWNLOAD_ROOT/data" --include 'contact/*' 'isaac51/*' --max-workers 8
"$(dirname "$PYTHON")/modelscope" download --model Qwen/Qwen3.5-2B --local_dir "$DOWNLOAD_ROOT/models/Qwen3.5-2B" --max-workers 8
