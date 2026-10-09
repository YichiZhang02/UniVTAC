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

```bash
CUDA_VISIBLE_DEVICES=0 bash train_encoder.sh \
  --method ResNet --size S --input marker_rgb

CUDA_VISIBLE_DEVICES=1 bash train_encoder.sh \
  --method ResNet --size S --input depth_deform
```

`--method` 支持 `ResNet / VAE / MAE / DINOv2 / I-JEPA / V-JEPA`。权重和日志保存到 `encoder_results/<input>/<method>/<size>/<时间>/`。

### Policy

全部任务、全部 episode、双相机、四卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
POLICY=starvla_groot TRAINING_MODE=multi_task TASKS='' EPISODES=0 CAMERAS=all \
TACTILE_MODE=encode TACTILE_INPUT_MODE=marker_rgb TACTILE_TYPE=cls \
TACTILE_INSERT_LOCATION=encoder ENCODER_METHOD=ResNet ENCODER_SIZE=S \
STEPS=30000 BATCH_SIZE=64 CHUNK_SIZE=32 SAVE_FREQ=5000 \
bash train_policy.sh
```

`BATCH_SIZE=64` 为全局 batch，四卡时单卡为 16。`POLICY` 支持 `act / pi05 / starvla_groot`；触觉可换为 `depth_deform`、`full`。默认选择对应 Encoder 的最新 `encoder.pth`，也可设置 `ENCODER_CKPT=/path/to/encoder.pth`。


配置、checkpoint 和日志统一保存在 `policy_results/` 的实体实验目录；权重文件为 `checkpoint_<step>.pt`，不使用软链接。更多参数见 [policy/TRAINING.md](policy/TRAINING.md)。

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
git commit -m "Update policy evaluation and documentation"
git push origin isaac51
```

`git add -A` 会暂存所有未忽略的改动；提交前可用 `git status --short` 核对文件清单。`policy_results/` 和 `test_results/` 中的模型、日志和视频不会上传，各目录的 `.gitkeep` 会保留。若推送时提示远端有新提交，先拉取并解决可能出现的冲突，再重新推送。
