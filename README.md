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

**测试：5090。** 使用独立环境，需要 Isaac Sim / Isaac Lab / TacEx / cuRobo；安装流程待验证。`requirements_train.txt` 仅用于训练。

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

```bash
RUN_DIR=/path/to/policy_results/experiment

# 直接使用实际 checkpoint 文件。
bash test_policy.sh --checkpoint "$RUN_DIR/checkpoint_30000.pt" --inspect

# 训练环境：加载权重，对本地 HDF5 单帧推理。
bash test_policy.sh --checkpoint "$RUN_DIR" --task lift_can --offline

# 测试环境：模拟器评测；5090 上仍待验证。
bash test_policy.sh --checkpoint "$RUN_DIR" \
  --task lift_can --task-config demo --gpu 0 --headless
```

默认加载目录中最新 checkpoint，可用 `--step 30000` 指定步数。转移权重时保留同目录的 `train_config.yml`、`dataset_stats.json` 和 `episodes.json`；离线推理不代表模拟器任务成功率。

当前仅保留 GelSight Mini 传感器资产，包含标定权重和数组，位于 `resources/third_party/TacEx/source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/`，随 Git 仓库提供。
