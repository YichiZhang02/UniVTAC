# StarVLA PI v3

从 `/mnt/yichi_data/starVLA` 迁入 QwenPI_v3，沿用 `starvla_groot` 的 vendor 布局及 UniVTAC 数据接口。上游版本见 `vendor/UPSTREAM_COMMIT`，许可证见 `vendor/LICENSE`。

当前共享配置使用 Qwen3.5-2B：24 层语言骨干、2048 维隐藏特征。每层使用独立 LayerNorm + Linear 压缩到 1024，接入 24 层联合注意力 DiT。每层 Q 来自经过时间条件 AdaNorm 的动作特征；K/V 来自该层 1024 维条件特征与当前动作特征的拼接，使用该层共享的 K/V 投影。动作 tokens 双向可见，条件 padding 被屏蔽。不使用 VLM 内部 KV cache。

Action head 隐藏维度固定为 **1024**，16 个注意力头、每头 64 维。Future tokens 数量为 **0**，默认 DiT Query 序列仅有 50 个动作 tokens。8 维状态量化后通过指令文本输入。输出为 50×8，默认执行前 30 步。PI 和 GR00T 的默认推理积分步数均为 10，由 `num_inference_timesteps` 控制，与训练的 `flow_repeats` 无关。

触觉支持 `vlm` / `head`，默认 `vlm`。`vlm` 模式以 V→T→L 注入触觉；每层完整 VTL hidden states（包括启用时的包裹位置）使用该层投影压缩到 1024，直接作为对应 DiT 层的条件序列。没有单独提取或追加最后一层 T latent 的分支。`head` 模式下触觉绕过 VLM，编码后直接作为 head 条件。三种包裹、三种 RoPE 和三种标准 attention mask 可独立配置。

Qwen3.5 的 18 个 GatedDeltaNet 保持原实现；mask 和触觉 RoPE 选项只作用于 6 个标准 attention 层。详细配置、mask 作用范围及检查命令见 `training/README.md` 的 StarVLA tactile fusion。

训练示例（未启动正式训练）：

```bash
bash training/launch.sh --model starvla_pi --output runs/starvla_pi_vision
bash training/launch.sh --model starvla_pi --output runs/pi_tactile \
  --tactile --encoder rae --rae-variant rae-4in1 --tokens pool3x3 \
  --tactile-fusion vlm --tactile-wrapper tactile \
  --tactile-rope 3d --vlm-attention-mask modality
```

检查点通过 `training.inference.ChunkPolicy` 和现有 `policy/fresh` 接口加载；DLC 打包入口也接受 `--model starvla_pi`。新增触觉融合参数要求匹配的检查点配置和权重。旧 PI 检查点若含 `[32, 1024]` 的 future tokens，不能直接严格加载；旧全 cross-attention 权重虽可保持形状，注意力语义已变化，仍需重新训练或验证。
