#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_DATASET_ID="${SOURCE_DATASET_ID:-byml2024/UniVTAC}"
DOWNLOAD_REVISION="${DOWNLOAD_REVISION:-master}"
UNIVTAC_RAW_DIR="${UNIVTAC_RAW_DIR:-$REPO_ROOT/resources/data}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"

usage() {
    cat <<'EOF'
用法：bash scripts/download_data.sh [选择器...] [选项]

选择器（至少一个，可组合）：
  --task [TASK]          下载全部任务或指定任务；使用时必须提供 --version
  --contact [SHAPE]      下载全部接触形状或指定形状
  --checkpoint [TASK]    下载全部 checkpoint 或指定任务的 checkpoint

选择器可以重复，例如：
  --task lift_can --task insert_hole --version 51

示例：
  bash scripts/download_data.sh --task --version 51
  bash scripts/download_data.sh --task lift_can --version 45
  bash scripts/download_data.sh --contact
  bash scripts/download_data.sh --checkpoint
  bash scripts/download_data.sh --task --version 51 --contact --checkpoint

选项：
  --version 45|51        task 数据对应的 Isaac Sim 版本
  --revision REVISION    数据集 revision，默认 master
  --output DIR           下载根目录，默认 resources/data/
  --workers N            并行下载 worker 数，默认 8
  --force                强制重新下载
  -h, --help             显示帮助

下载后保留仓库目录结构：
  <output>/isaac45/<task>/{hdf5/*.hdf5,metadata.json}
  <output>/isaac51/<task>/{hdf5/*.hdf5,metadata.json}
  <output>/contact/<shape>/{hdf5/*.hdf5,metadata.json}
  <output>/checkpoints/...
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "缺少命令：$1。请在当前 Python 环境中安装 modelscope。" >&2
        exit 1
    fi
}

die_download() {
    echo "download 参数错误：$*" >&2
    echo "运行 bash scripts/download_data.sh --help 查看用法。" >&2
    exit 2
}

validate_name() {
    local kind="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        die_download "$kind 名称不合法：$value"
    fi
}

validate_task_name() {
    local value="$1"
    case "$value" in
        grasp_classify|insert_HDMI|insert_hole|insert_tube|lift_bottle|lift_can|pull_out_key|put_bottle_in_shelf) ;;
        *) die_download "未知 task：$value" ;;
    esac
}

download() {
    local task_selected=0
    local task_all=0
    local contact_selected=0
    local contact_all=0
    local checkpoint_selected=0
    local checkpoint_all=0
    local version=""
    local revision="$DOWNLOAD_REVISION"
    local output_dir="$UNIVTAC_RAW_DIR"
    local workers="$DOWNLOAD_WORKERS"
    local force=0
    local -a task_names=()
    local -a contact_shapes=()
    local -a checkpoint_tasks=()
    local -a include_patterns=("README.md" "dataset_infos.json")
    local -a force_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --task)
                task_selected=1
                if [[ $# -gt 1 && "$2" != -* ]]; then
                    task_names+=("$2")
                    shift
                else
                    task_all=1
                fi
                ;;
            --task=*)
                task_selected=1
                task_names+=("${1#*=}")
                ;;
            --contact)
                contact_selected=1
                if [[ $# -gt 1 && "$2" != -* ]]; then
                    contact_shapes+=("$2")
                    shift
                else
                    contact_all=1
                fi
                ;;
            --contact=*)
                contact_selected=1
                contact_shapes+=("${1#*=}")
                ;;
            --checkpoint|--checkpoints)
                checkpoint_selected=1
                if [[ $# -gt 1 && "$2" != -* ]]; then
                    checkpoint_tasks+=("$2")
                    shift
                else
                    checkpoint_all=1
                fi
                ;;
            --checkpoint=*|--checkpoints=*)
                checkpoint_selected=1
                checkpoint_tasks+=("${1#*=}")
                ;;
            --version)
                [[ $# -gt 1 ]] || die_download "--version 缺少参数"
                version="$2"
                shift
                ;;
            --version=*) version="${1#*=}" ;;
            --revision)
                [[ $# -gt 1 ]] || die_download "--revision 缺少参数"
                revision="$2"
                shift
                ;;
            --revision=*) revision="${1#*=}" ;;
            --output)
                [[ $# -gt 1 ]] || die_download "--output 缺少参数"
                output_dir="$2"
                shift
                ;;
            --output=*) output_dir="${1#*=}" ;;
            --workers)
                [[ $# -gt 1 ]] || die_download "--workers 缺少参数"
                workers="$2"
                shift
                ;;
            --workers=*) workers="${1#*=}" ;;
            --force) force=1 ;;
            -h|--help)
                usage
                return
                ;;
            *) die_download "未知参数：$1" ;;
        esac
        shift
    done

    if (( ! task_selected && ! contact_selected && ! checkpoint_selected )); then
        die_download "至少指定 --task、--contact、--checkpoint 中的一项"
    fi
    if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
        die_download "--workers 必须是正整数：$workers"
    fi

    if (( task_selected )); then
        case "$version" in
            45|4.5|isaac45) version="45" ;;
            51|5.1|isaac51) version="51" ;;
            "") die_download "下载 task 时必须指定 --version 45 或 --version 51" ;;
            *) die_download "--version 仅支持 45 或 51：$version" ;;
        esac
        if (( task_all )) || [[ ${#task_names[@]} -eq 0 ]]; then
            include_patterns+=(
                "isaac${version}/*/hdf5/*.hdf5"
                "isaac${version}/*/metadata.json"
            )
        else
            local task
            for task in "${task_names[@]}"; do
                validate_task_name "$task"
                include_patterns+=(
                    "isaac${version}/${task}/hdf5/*.hdf5"
                    "isaac${version}/${task}/metadata.json"
                )
            done
        fi
    elif [[ -n "$version" ]]; then
        die_download "--version 只在下载 task 时使用"
    fi

    if (( contact_selected )); then
        if (( contact_all )) || [[ ${#contact_shapes[@]} -eq 0 ]]; then
            include_patterns+=("contact/*/hdf5/*.hdf5" "contact/*/metadata.json")
        else
            local shape
            for shape in "${contact_shapes[@]}"; do
                validate_name "contact shape" "$shape"
                include_patterns+=(
                    "contact/${shape}/hdf5/*.hdf5"
                    "contact/${shape}/metadata.json"
                )
            done
        fi
    fi

    if (( checkpoint_selected )); then
        if (( checkpoint_all )) || [[ ${#checkpoint_tasks[@]} -eq 0 ]]; then
            include_patterns+=("checkpoints/*")
        else
            include_patterns+=("checkpoints/encoder.pth")
            local checkpoint_task
            for checkpoint_task in "${checkpoint_tasks[@]}"; do
                validate_task_name "$checkpoint_task"
                include_patterns+=("checkpoints/${checkpoint_task}/*")
            done
        fi
    fi

    if (( force )); then
        force_args+=("--force")
    fi
    require_command modelscope
    mkdir -p "$output_dir"
    echo "数据集：$SOURCE_DATASET_ID@$revision"
    echo "下载目录：$output_dir"
    printf '包含路径：\n'
    printf '  %s\n' "${include_patterns[@]}"
    modelscope download "$SOURCE_DATASET_ID" \
        --repo-type dataset \
        --revision "$revision" \
        --local-dir "$output_dir" \
        --include "${include_patterns[@]}" \
        --max-workers "$workers" \
        "${force_args[@]}"
}

# Keep the former `download` subcommand as a compatibility alias while making
# the script itself a single-purpose downloader.
if [[ "${1:-}" == "download" ]]; then
    shift
fi
download "$@"
