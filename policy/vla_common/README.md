# VLA 共用代码

这里保存 PI0.5 与 StarVLA-GR00T 共用的基础代码；模型本身分别在 [`../pi05/`](../pi05/) 和 [`../starvla_groot/`](../starvla_groot/)。

| 路径 | 作用 |
| --- | --- |
| `engine/configs/` | feature 类型和共用配置 |
| `engine/optim/` | 优化器与学习率调度 |
| `engine/processor/` | 共用 processor |
| `engine/utils/` | 通用工具 |
| `pretrained.py` | pretrained policy 基类 |
| `sensor_routing.py` | 相机、状态、动作和触觉路由 |
| `tactile_encode.py` | 本仓库 encoder checkpoint 适配 |
| `tactile_temporal_processor.py` | 触觉时间窗口处理 |
| `pi_gemma.py` | PI0.5 的 Gemma/PaliGemma 支持代码 |

训练与测试入口见 [`../TRAINING.md`](../TRAINING.md)。这些代码由 StarVTLA 中的 LeRobot / OpenPI 相关实现适配，保留源文件的版权与许可声明。
