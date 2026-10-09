# UniVTAC Quick Start

以下命令均在仓库根目录执行。

## 1. 环境安装

**训练：A100/A800。** 参考当前 A100 环境：Python 3.11、PyTorch 2.7.0、CUDA 12.6。

```bash
conda create -n univtac python=3.11 -y
conda activate univtac
conda install -c nvidia cuda-toolkit=12.6 -y
python -m pip install -r requirements_train.txt
export PYTHON_BIN="$(command -v python)"
```

**测试：5090。** 使用独立的 `univtac` 环境，已安装 Isaac Sim / Isaac Lab / TacEx / cuRobo，并通过两种触觉后端的无窗口测试。`requirements_train.txt` 仅用于训练。

RTX 5090 + Isaac Sim 5.1 的逐步命令和验收顺序见 [手动安装与验证](README_5090.md)。

## 2. 数据与预训练模型下载

数据、预训练模型及训练结果目录在 Git 中只保留 `.gitkeep`；本地内容另行下载或传输。

### 训练数据

```bash
python -m pip install modelscope==1.40.1

# Encoder 数据 → resources/data/contact/
bash scripts/download_data.sh --contact

# Policy 数据：全部 Isaac Sim 5.1 任务 → resources/data/isaac51/
bash scripts/download_data.sh --task --version 51
```

### 预训练模型

下载 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 和 [PI0.5](https://huggingface.co/lerobot/pi05_base)：

```bash
hf download Qwen/Qwen3.5-2B \
  --local-dir resources/pretrained_models/Qwen3.5-2B

hf download lerobot/pi05_base \
  --local-dir resources/pretrained_models/pi05_base

# PI0.5 还需要本地 PaliGemma tokenizer。
hf download google/paligemma-3b-pt-224 \
  --include tokenizer.json tokenizer_config.json config.json \
  --local-dir resources/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer
```

下载 tokenizer 前，需在 [PaliGemma 页面](https://huggingface.co/google/paligemma-3b-pt-224)接受访问条款。

## 3. 训练 Encoder 和 Policy

### Encoder

- 方法（`--method`）：`ResNet / VAE / MAE / DINOv2 / I-JEPA / V-JEPA`。
- 尺寸（`--size`）：`S / B`，默认 `S`。
- 触觉表示（`--input`）：`marker_only / rgb_only / marker_rgb / depth_deform`，默认 `marker_rgb`。

```bash
CUDA_VISIBLE_DEVICES=0 bash train_encoder.sh \
  --method ResNet --size S --input marker_rgb
```

权重和日志保存到 `encoder_results/<input>/<method>/<size>/<时间>/`。

### Policy

- 模型（`POLICY`）：`act`（ACT）、`pi05`（PI0.5）、`starvla_groot`（StarVLA-GR00T）。
- 触觉模式（`TACTILE_MODE`）：`none / as_image / encode`。
- 表示方式（`TACTILE_INPUT_MODE`）：`marker_only / rgb_only / marker_rgb / depth_deform`，与 Encoder 相同。
- `encode` 模式：`ENCODER_METHOD` 和 `ENCODER_SIZE` 与上述 Encoder 选项相同；`TACTILE_TYPE=cls / full` 分别使用全局特征或完整 token 序列，`TACTILE_INSERT_LOCATION=encoder / decoder` 指定插入位置。

全部任务、全部 episode、双相机、四卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 \
POLICY=starvla_groot TRAINING_MODE=multi_task TASKS='' EPISODES=0 CAMERAS=all \
TACTILE_MODE=encode TACTILE_INPUT_MODE=marker_rgb \
TACTILE_TYPE=full TACTILE_INSERT_LOCATION=encoder ENCODER_METHOD=VAE ENCODER_SIZE=S \
STEPS=100_000 BATCH_SIZE=32 CHUNK_SIZE=32 SAVE_FREQ=10_000 \
bash train_policy.sh
```

`BATCH_SIZE=64` 为全局 batch，四卡时单卡为 16。默认选择对应 Encoder 的最新 `encoder.pth`，也可设置 `ENCODER_CKPT=/path/to/encoder.pth`。

## 4. 测试 Policy

### 多任务模拟器评测

`test_policy.sh` 的前四个位置参数依次是 `MODEL_ID`、checkpoint 的 `STEP`、可用 GPU 编号列表和并行 GPU 数。省略参数时使用脚本内的默认值；当前默认为下面这个模型的第 30000 步，在 0–7 卡上并行评测 8 个 task。每张卡一次运行一个 task，完成后接手队列中的下一个。

```bash
# 先查看将运行的 task、GPU 和参数；不启动模拟器。
bash test_policy.sh --dry-run

# 使用脚本中的默认模型、步数和 8 张 GPU。
bash test_policy.sh

# 只使用空闲的 0、2、5 卡，最多同时运行 3 个 task。
bash test_policy.sh \
  20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder \
  30000 0,2,5 3
```

当前脚本使用 `demo` 任务配置、`seen` 指令，每个 task 目标为 **20 个有效回合**；起始 seed 为 `1000000`，之后递增，不设置最大 seed。每个 task 遇到 10 次评测异常会停止，reset 超时设为 600 秒。批量评测使用 headless 模式并关闭 livestream。需要调整这些参数时，修改 [test_policy.sh](test_policy.sh) 顶部的设置。

结果保存在 `test_results/<MODEL_ID>/<STEP>/`：`logs/` 是各 task 的控制台日志，`videos/<task>/video/` 是回合视频，`results/` 是逐 task 统计，`summary.md` 和 `summary.json` 汇总成功率及各 task 成功率的算术平均值。若有 task 未完成目标回合数，平均值显示为 N/A。重测同一个模型和 step 时，脚本会把旧结果目录改名为 `_previous_<时间>` 后再启动，保留中断运行的日志和视频。

更多测试选项见 [policy/TRAINING.md](policy/TRAINING.md)。

### 推理资源与权重转移

使用 `--checkpoint` 指定训练目录时，默认加载目录中最新 checkpoint，可用 `--step 30000` 指定步数。转移权重时保留同目录的 `train_config.yml`、`dataset_stats.json`、`episodes.json` 和 `inference_assets/`；离线推理不代表模拟器任务成功率。

新训练的 starvla_groot 和 π0.5 会自动保存推理资源，并在 `train_config.yml` 中记录相对路径。starvla_groot 从目录内的 Qwen 配置创建模型，全部权重来自 policy checkpoint；π0.5 从目录内加载 PaliGemma tokenizer。推理无需原始 pretrained model 或 tactile encoder 目录。未转换的旧训练目录仍支持原来的加载方式。

已有训练结果可直接补齐资源，不重写 `.pt` 权重文件；原配置备份为 `train_config.yml.pre-inference.bak`：

```bash
python scripts/prepare_inference_checkpoints.py --root policy_results          # 预览
python scripts/prepare_inference_checkpoints.py --root policy_results --apply  # 转换
```

也可用 `--root "$RUN_DIR"` 转换单个训练目录，用 `--models-root PATH` 指定转换时的预训练资源目录。同一训练目录的所有 step 共用一份资源，移动整个目录后相对路径仍有效。

当前仅保留 GelSight Mini 传感器资产，包含标定权重和数组，位于 `resources/third_party/TacEx/source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/`，随 Git 仓库提供。

## 5. Git 使用

本仓库的 GitHub 远端为 `origin`，当前开发分支为 `isaac51`。

### 从 GitHub 拉取

在 `isaac51` 分支上，先确认没有未提交的改动，再拉取远端更新：

```bash
git status -sb
git pull --ff-only origin isaac51
```

如果本地已有尚未推送的提交，`--ff-only` 无法合并时可使用 `git pull --rebase origin isaac51`。有未提交改动时，先提交或暂存这些改动，再拉取。

### 提交并推送到 GitHub

提交前检查改动，然后推送：

```bash
git status -sb
git diff --check
git add -A
git diff --cached --stat
git diff --cached --check
git commit -m "..."
git push origin isaac51
```

`git add -A` 会暂存所有未忽略的改动；提交前可用 `git status --short` 核对文件清单。`policy_results/` 和 `test_results/` 中的模型、日志和视频不会上传，各目录的 `.gitkeep` 会保留。若推送时提示远端有新提交，先拉取并解决可能出现的冲突，再重新推送。
