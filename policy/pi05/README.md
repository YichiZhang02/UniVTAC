# PI0.5

PI0.5 的模型实现位于 `policy/pi05/`。PaliGemma 编码图像和任务文本，Gemma action expert 通过 flow matching 生成动作序列；共用配置、优化器、触觉编码和 processor 在 `policy/vla_common/`。

## 环境与模型

环境安装遵循根目录 [README_yichi.md](../../README_yichi.md) 的 A100/A800 训练 Quick Start，统一使用 `requirements_train.txt`。5090 模拟器测试环境仍待验证。

本地基础权重与 tokenizer 放在：

```text
resources/pretrained_models/pi05_base/
├── config.json
├── model.safetensors
└── paligemma-3b-pt-224-tokenizer/
```

## 训练与测试

从仓库根目录运行：

```bash
POLICY=pi05 TACTILE_MODE=none bash train_policy.sh

bash test_policy.sh --checkpoint /path/to/run --inspect
bash test_policy.sh --checkpoint /path/to/run --task lift_can --offline
```

当前统一训练使用绝对 8D 关节 state/action，动作从 `t+1` 开始；默认 chunk 为 32。触觉支持 `none`、`as_image`、`encode`，多卡训练使用 `NUM_PROCESSES`，`BATCH_SIZE` 是全局 batch。完整参数和 checkpoint 约定见 [TRAINING.md](../TRAINING.md)。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `configuration_pi05.py` | 模型配置与输入输出契约 |
| `modeling_pi05.py` | 模型、训练 loss 和动作推理 |
| `processor_pi05.py` | processor 定义 |
| `base_config.yml` | 模型与优化器默认配置 |

该实现由 StarVTLA 中的 PI0.5 代码适配，源代码保留 Physical Intelligence 和 Hugging Face 的版权及许可声明。
