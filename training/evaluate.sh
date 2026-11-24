#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="$(realpath "${1:?Usage: evaluate.sh CHECKPOINT [EPISODES]}")"
EPISODES="${2:-50}"
POLICY_PORT="${POLICY_PORT:-5557}"
START_SEED="${START_SEED:-1000000}"
MAX_SEED="${MAX_SEED:--1}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/mnt/yichi_data/miniconda3/envs/univtac/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-/mnt/yichi_data/miniconda3/envs/UniVTAC/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-$(dirname "$CHECKPOINT")/evaluation}"
mkdir -p "$EVAL_ROOT"
export CHECKPOINT
POLICY_SOURCE="$(dirname "$CHECKPOINT")/source"
if [[ ! -d "$POLICY_SOURCE" ]]; then POLICY_SOURCE="$PWD"; fi
export PYTHONPATH="$POLICY_SOURCE:$POLICY_SOURCE/policy/starvla_groot/vendor:$PWD${PYTHONPATH:+:$PYTHONPATH}"
(cd "$POLICY_SOURCE" && exec env CUDA_VISIBLE_DEVICES="${POLICY_GPU:-1}" "$TRAIN_PYTHON" -m training.policy_server --checkpoint "$CHECKPOINT" --port "$POLICY_PORT") > "$EVAL_ROOT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
ready=0
for ((attempt=0; attempt<180; attempt++)); do
    if grep -q 'Policy server ready' "$EVAL_ROOT/server.log" 2>/dev/null; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then cat "$EVAL_ROOT/server.log"; exit 1; fi
    sleep 2
done
if [[ "$ready" != 1 ]]; then echo 'Policy server startup timed out'; exit 1; fi
export POLICY_SERVER="tcp://127.0.0.1:$POLICY_PORT"
for task in ${TASKS:-grasp_classify insert_HDMI insert_hole insert_tube lift_bottle lift_can pull_out_key put_bottle_in_shelf}; do
    RESULT_FILE="$EVAL_ROOT/$task.result.json"
    if "$TRAIN_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["complete"] and r["valid"]==int(sys.argv[2]) else 1)' "$RESULT_FILE" "$EPISODES" 2>/dev/null; then continue; fi
    mapfile -t TASK_OVERRIDES < <("$TRAIN_PYTHON" "$PWD/training/eval_config.py" "$task")
    if [[ "${#TASK_OVERRIDES[@]}" != 2 ]]; then echo "Missing evaluation limits for $task"; exit 1; fi
    if [[ -n "${SMOKE_ACTION_LIMIT:-}" ]]; then TASK_OVERRIDES+=("task_overrides.step_lim=$SMOKE_ACTION_LIMIT"); fi
    PYTHONPATH="$PWD:$PWD/third_party/TacEx/source/tacex_assets:$PWD/third_party/TacEx/source/tacex_uipc:$PWD/third_party/TacEx/source/tacex:$PWD/third_party/TacEx/source/tacex_tasks${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="${SIM_GPU:-0}" "$SIM_PYTHON" scripts/eval_policy.py "$task" clean51 fresh/deploy \
        --result-file "$RESULT_FILE" --headless --start_seed "$START_SEED" --max_seed "$MAX_SEED" --total_num "$EPISODES" \
        --config-overrides "replay_settings.save_root_dir=$EVAL_ROOT" "${TASK_OVERRIDES[@]}" > "$EVAL_ROOT/$task.log" 2>&1
done
