# StarVLA-GR00T

StarVLA-GR00T 的模型实现位于 `policy/starvla_groot/`，由 Qwen 视觉语言模型和 GR00T flow-matching DiT action head 组成。共用配置、优化器、触觉编码和 processor 在 `policy/vla_common/`。

## 环境与模型

环境安装遵循根目录 [README_yichi.md](../../README_yichi.md) 的 A100/A800 训练 Quick Start，统一使用 `requirements_train.txt`。统一训练入口默认读取 `resources/pretrained_models/Qwen3.5-2B/`。5090 模拟器测试环境仍待验证。

## 训练与测试

从仓库根目录运行：

```bash
POLICY=starvla_groot TRAINING_MODE=multi_task EPISODES=0 CAMERAS=all \
  TACTILE_MODE=encode TACTILE_INPUT_MODE=marker_rgb \
  ENCODER_METHOD=ResNet ENCODER_SIZE=S TACTILE_TYPE=cls \
  TACTILE_INSERT_LOCATION=encoder bash train_policy.sh

bash test_policy.sh --checkpoint /path/to/run --inspect
bash test_policy.sh --checkpoint /path/to/run --task lift_can --offline
```

当前统一训练使用绝对 8D 关节 state/action，动作从 `t+1` 开始；默认 chunk 为 32。触觉支持 `none`、`as_image`、`encode`。`encode` 使用本仓库训练的六种 encoder，支持 `cls/full` 和 `encoder/decoder` 插入位置。

多卡训练使用 `NUM_PROCESSES`，`BATCH_SIZE` 是全局 batch；四个八卡机器的批量启动脚本仍位于 `scripts/`。完整参数和 checkpoint 约定见 [TRAINING.md](../TRAINING.md)。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `configuration_starvla_groot.py` | 模型配置与输入输出契约 |
| `modeling_starvla_groot.py` | 模型、训练 loss 和动作推理 |
| `qwen_vl_interface.py` | Qwen 多模态接口 |
| `action_head/` | GR00T DiT 动作头 |
| `processor_starvla_groot.py` | processor 定义 |
| `base_config.yml` | 模型与优化器默认配置 |

该实现由 StarVTLA / starVLA 代码适配，源代码保留相关版权及许可声明。
