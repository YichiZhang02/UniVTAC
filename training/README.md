# Isaac51 fresh reproduction

Upstream UniVTAC branch `isaac51`, commit `d541e5568227ca3b66104d294f63c80acad7c52c`.
StarVLA source is vendored under `policy/starvla_groot/vendor`; its exact revision is in `UPSTREAM_COMMIT` and its license is retained.
Only the legacy RAE/RAEv5 encoder architecture was consulted. All data and weights were downloaded into `/mnt/yichi_data/UniVTAC_downloads` afresh.

## Data and action contract

`training/config.json` is shared by both baselines. All 800 Isaac51 episodes from all eight tasks are used for training; no validation/test episodes are held out. Every timestep except the terminal state is an anchor. The first eight entries of `embodiment/joint` are used, matching upstream policies; the ninth redundant gripper entry is omitted.

`relative_joint` uses `q[t+k+1, :7] - q[t, :7]` for the seven arm joints and absolute `q[t+k+1, 7]` for the gripper, shared by every policy in this training entry point. Statistics use the same mixed action space. Chunks contain 50 targets and terminal actions are held when fewer than 50 remain. ACT masks padded targets in its loss; the migrated GR00T head learns terminal holds. At inference, only the seven arm coordinates have the anchor added back, and only the first 30 targets are returned. New checkpoints record `action_contract=relative_arm_absolute_gripper_v1`. Historical checkpoints without this field retain their original all-relative decoding; new training requires newly computed statistics and cannot reuse those weights as an equivalent model. The encoded image channels follow the upstream OpenCV encode/decode roundtrip without an extra channel swap.

StarVLA uses `action_attention_dim=1024` (16 attention heads of 64 dimensions), 16 alternating cross/self-attention blocks, and 1024-wide state/action decoder MLPs. `action_future_tokens=0` removes the auxiliary future tokens: the head sequence contains one state token and 50 action tokens (51 total). Historical configurations without `action_attention_dim` retain the original DiT-B width of 768; those without `action_future_tokens` retain 32 future tokens when reloading their checkpoints.

## Train

```bash
# Default: four GPUs, micro batch16, global batch64, 50,000 optimizer updates.
bash training/launch.sh --model act --output runs/act_vision
bash training/launch.sh --model starvla --output runs/starvla_vision
# Local two-GPU smoke; global batch remains64 via accumulation2.
GPUS=2 bash training/launch.sh --model act --output runs/local_smoke --steps 1
# DLC: source is frozen and hashed before submission.
python -m training.submit_dlc --model act --gpus 4
python -m training.submit_dlc --model starvla --gpus 4
```

The DLC endpoint must be `pai-dlc.cn-wulanchabu.aliyuncs.com`. Passing only `--region` leaves the CLI's default Shanghai endpoint in place and causes misleading authorization errors.

## Tactile options

All new tactile-enabled runs use `depth_deform_radial`, shared by contact pretraining, ACT/StarVLA datasets, and the live rollout adapter. This is derived from raw camera depth and paired reference/current marker coordinates, not reconstructed from RGB pixels. R = clip((30 - camera_depth_mm)/5, 0, 1); GB = 0.5 + 0.5 * displacement / max(30, L2(displacement)). Marker displacement is interpolated in reference coordinates, in original sensor pixels. `marker_full` is preferred when present; duplicate reference points within 0.001 px are averaged.

The default interpolation is `tps_bilinear`, grid 28: solve TPS at marker controls, evaluate only a 28x28 grid, then use exact bilinear upsampling to 224x224. `linear` implements triangular linear interpolation with affine + IDW residual extrapolation outside the convex hull. Full `tps` remains available as a reference. Contact-only calibration on 56 high-motion frames is saved in `artifacts/tactile_interpolation/metrics.json`: dense TPS 50.19 ms/frame; linear 26.63 ms/frame and 0.898 px mean disagreement; TPS28 + bilinear 2.26 ms/frame and 0.0255 px mean disagreement (0.000257 encoded GB MAE). These are single-thread CPU measurements excluding I/O, not ground-truth reconstruction errors or GPU speed comparisons.

Add `--tactile --encoder univtac` to use the native ResNet trunk, or `--tactile --encoder rae --rae-variant rae-4in1` for ViT-S (384 dimensions,12 blocks,6 heads,16x16 patches). Add `--rae-size B` for ViT-B (768 dimensions,12 blocks,12 heads); ViT-S remains the default. All RAE variants use learned absolute spatial/time positions, no RoPE. `rae-1in1` processes one frame independently (196 patch tokens); `rae-4in1` jointly encodes four frames and compresses time using Conv1d AFTER the complete encoder (196 patch tokens). `pool3x3` returns 9 tokens per latent frame; `cls` returns one actual encoder CLS per sensor. Four-frame histories use stored-frame stride 2, chronological order, and earliest-frame padding. The rae-1in1 policy consumes only the current frame. The current policy wrapper supports both sensors.

`encoder_checkpoint` in config loads a strictly matched, freshly trained encoder. A null value initializes a new encoder; it does not silently use an old checkpoint. Native UniVTAC `cls` denotes globally pooled convolution features.

```bash
# Epoch-based training; use --size B for the matching ViT-B encoder.
python -m torch.distributed.run --standalone --nproc_per_node=4 -m training.pretrain_rae --variant rae-1in1 --size S --epochs 100 --output runs/rae1_contact
python -m torch.distributed.run --standalone --nproc_per_node=4 -m training.pretrain_rae --variant rae-4in1 --size S --epochs 100 --output runs/rae4in1_contact
```

RAE contact pretraining uses all 638 local contact episodes, both sensors (353,690 sensor anchors), with four frames at stored-frame stride 2. Each variant reconstructs the same four radial input frames with pixel MSE; rae-1in1 processes them independently, rae-4in1 expands its compressed latent back to four frames. Only encoder weights are needed by policies. Default global batch is 64 four-frame clips, micro batch 4 per rank with gradient accumulation, AdamW lr1e-4, 100 full epochs. Every training anchor is used; DistributedSampler pads at most world_size-1 duplicate anchors to give ranks equal lengths, and partial last accumulation groups are weighted correctly. Validation runs after every epoch; minimum val MSE selects the best epoch. The test split is evaluated once after training on the selected best weights. `last.pt` supports automatic epoch-boundary recovery. `best.pt` stores encoder/decoder state dictionaries, optimizer, architecture and configuration; `best_model.pt` stores the complete CPU model object. `architecture.json` and `model_source/encoder/rae.py` retain the exact reconstructable architecture. Four ViT-S/B × 1in1/4in1 jobs and their immutable source/evaluation snapshots are recorded under `runs/rae_100epoch_20261116`.

`data/encoder_eval_v1/raw_clips.h5` is the frozen shared encoder evaluation archive (84 MiB), with independent copies of original encoded marker RGB, camera depth and paired marker coordinates. `manifest.json` records sources, indices, motion descriptors and checksums; `val.json` / `test.json` define disjoint source episodes. It contains 20 clips per policy task, 160 clips / 640 sensor frames total, each four frames at stride 2. Both sensors are balanced. Val/test each contain 80 clips / 320 frames, 10 clips per task. Each clip uses one sensor, not a synchronized two-sensor pair. Single-frame models process all 640 frames independently; multi-frame models process 160 clips. No converted inputs are stored as evaluation ground truth: conversion uses the checkpoint's representation specification at evaluation time.

Selection ranks each episode's maximum temporal marker-displacement change, restricts to the top 65% motion pool, then selects diverse motion descriptors from distinct episodes. This is a deliberately high-change stress set, not an unbiased estimate of policy-task frame distribution. It is disjoint from contact pretraining, but sourced from episodes included in policy training; it must not be reported as a held-out policy success benchmark. Preview and checks are saved as `preview.png` and `verification.json`.

```bash
# Refuses to overwrite an existing frozen archive.
python -m training.build_encoder_eval
python -m training.verify_encoder_eval
python -m training.benchmark_interpolation
python -m training.test_tactile_rae
# Explicit checkpoint reconstruction evaluation, when trained weights exist:
python -m training.evaluate_rae --checkpoint runs/rae4in1_contact/step_050000.pt --split test --output runs/rae4in1_contact/test.json
```

## Verification and inference

```bash
python -m training.verify_downloads
python -m training.contract_test
python -m training.smoke --model act
python -m training.smoke --model starvla
python -m training.smoke --model rae
python -m training.smoke --model act --tactile --encoder rae --tokens pool3x3
```

`training.inference.ChunkPolicy` loads a checkpoint with its own configuration and normalization. Its `predict` accepts raw eight-coordinate `state`, camera tensors `[B,2,3,224,224]` in `[0,1]`, instruction list `lang`, and optional tactile `[B,2,4,3,224,224]`, and returns absolute joint targets `[B,30,8]`.

Training smoke tests establish software execution, not reproduced success rates. A completed 50k checkpoint and simulator rollouts are required before reporting a baseline result. The simulator adapter and evaluation launcher are provided; success rates must be measured with completed checkpoints.

The common simulator adapter is `policy/fresh/deploy.yml`. Set `CHECKPOINT` to the saved checkpoint and pass that deployment configuration to the upstream evaluation script. It consumes one of the 30 cached targets per simulator control tick, refreshes tactile history at each tick, and replans when the chunk is exhausted.

The flow head's number of noise repeats is explicitly saved as `flow_repeats` (default8). This setting changes the number of flow noise samples per demonstration, not the global demonstration batch64. The formal StarVLA run uses the upstream default8, verified with local micro batch16.

`finish_setup.py` waits for all 800 task episodes and both local DDP smoke checkpoints, validates the downloaded files, and submits ACT and StarVLA exactly once. It then monitors their DLC statuses in `runs/status.json`. Its submission marker is `runs/baseline_submissions_complete.json`; it reports failed prerequisites instead of submitting a known-broken training job.

Evaluate all eight tasks with 50 valid rollouts each:

```bash
bash training/evaluate.sh runs/<run>/step_050000.pt 50
```

The evaluation launcher uses GPU1 for a local policy server in the training environment, and GPU0 for Isaac Sim5.1 in the existing `UniVTAC` environment. `TRAIN_PYTHON`, `SIM_PYTHON`, `POLICY_GPU` and `SIM_GPU` can be overridden. This separates the VLM and simulator package requirements. Evaluation logs and videos go beside the checkpoint. This launcher has not yet been validated with simulator rollouts.

The fresh StarVLA optimizer uses the upstream learning-rate distinction: VLM1e-5 and action head1e-4. ACT uses1e-5. These values are saved in `config.json`.

Each formal run also records `environment.json` with Python, PyTorch, CUDA, GPU and the key package versions, alongside the training-file manifest, normalization statistics and immutable source snapshot.

PI v3 基线已迁入 `policy/starvla_pi`，action head 隐藏维度为 1024。使用 `bash training/launch.sh --model starvla_pi --output runs/starvla_pi_vision`，详见 `policy/starvla_pi/README.md`。

## StarVLA tactile fusion

New configurations default to `tactile_fusion=vlm`. Activate with `--tactile`.
The independent options are saved in the checkpoint configuration and supported by
`training.train`, `training.smoke`, and `training.submit_dlc`:

| CLI / config | Values | Default in config.json |
| --- | --- | --- |
| `--tactile-fusion` / `tactile_fusion` | `vlm`, `head` | `vlm` |
| `--tactile-wrapper` / `tactile_wrapper` | `none`, `vision`, `tactile` | `none` |
| `--tactile-rope` / `tactile_rope` | `none`, `1d`, `3d` | `1d` |
| `--vlm-attention-mask` / `vlm_attention_mask` | `causal`, `modality`, `bidirectional` | `causal` |

`full`, `pool3x3`, and `cls` token selection remains supported for both tactile
encoders. Tokens from both sensors are projected to Qwen's hidden width before
insertion between the last image's `vision_end` and the language instruction:
V → T → L. Chat-template control tokens remain in place. The template preamble
forms a preceding block; the image region, tactile region and language region
form the other three blocks.

`none` adds no delimiters. `vision` uses the pretrained `vision_start` and
`vision_end` embeddings around each sensor's tokens. `tactile` uses two learned
start/end embeddings around each sensor's tokens, without changing the tokenizer
vocabulary or pretrained embedding matrix. Delimiter hidden states remain in the complete VTL sequence passed to the head.

`none` RoPE uses zero coordinates for the inserted tactile region (identity
rotation), including delimiters. This only removes Qwen-level tactile RoPE;
the tactile encoder's own learned position embeddings remain in place.
`1d` uses increasing scalar positions repeated on all three M-RoPE axes, just
like language tokens. `3d` uses sensor-local (time,row,column) grids with the
same clock for both hands. Only this coordinate convention and wrapped sensor
grid extent follow UniVTAC1. Full tokens use the encoder's 14×14 RAE or 7×7
ResNet grid; pooled tokens use 3×3; CLS uses a single 1×1 position. The temporal
extent follows the encoder's **output** latent frames, not the raw history
length (the current four-frame RAE compresses to one latent frame). Image
coordinates are preserved, and subsequent language coordinates advance by the
inserted region's positional extent.

The VLM masks apply to Qwen3.5's six standard attention layers:

- `causal`: each token sees itself and preceding valid tokens.
- `modality`: tokens see all valid tokens within their block and preceding blocks.
- `bidirectional`: all valid V/T/L and template tokens are mutually visible.

As explicitly selected, all 18 GatedDeltaNet layers retain their original
causal computation and 2-D padding mask; they do not receive the 4-D mask and
have no M-RoPE. Thus `bidirectional` describes the standard-attention layers,
not a replacement of all hybrid mixers. The mask option also works with
head-only fusion and vision-only runs.

With default `vlm` fusion, tactile tokens enter Qwen in V→T→L order and
its complete VTL hidden sequence (including delimiters if enabled) becomes
the head's ordinary condition sequence:

- GR00T CA: Q = action; KV = the final VLM layer's complete VTL latents.
  SA still sees the complete state/action sequence. New configurations set
  `head_action_queries_only=true` so CA updates only action positions.
- PI: Q = action; KV = the corresponding VLM layer's complete VTL latents,
  projected to 1024, plus current action features. Each layer uses its own
  tactile latents as part of VTL; there is no separate final-layer T branch.
  Action-to-action visibility is bidirectional; condition padding is blocked.

`head` sends tactile encoder features directly to the action head without
injecting them into Qwen. Both training and inference call the same `encode`
implementation. There is no third fusion mode. Configs omitting
`tactile_fusion` now default to `vlm`; set `head` explicitly for older head-only
tactile weights. The fusion mode requires matching checkpoint weights.

Example (does not submit a job):

```bash
bash training/launch.sh --model starvla_pi --output runs/pi_tactile \
  --tactile --encoder rae --rae-variant rae-4in1 --tokens pool3x3 \
  --tactile-fusion vlm --tactile-wrapper tactile \
  --tactile-rope 3d --vlm-attention-mask modality
```

Synthetic tests:

```bash
python -m unittest training.test_tactile_vlm training.test_tactile_routing \
  policy.starvla_pi.tests.test_joint_attention -v
```

The unit suite covers all 27 wrapper/RoPE/mask combinations, RAE/ResNet grid
sizes, padding, causal equivalence to the native HF text model, checkpointed
gradients, complete per-layer VTL routing, and action-only GR00T CA with unchanged SA
sequence. These synthetic checks do not establish policy success rates.

## Sampled-anchor replacement run

The four original full-anchor DLC jobs were stopped at the user's request. `runs/rae_sample80k_100epoch_20261116/jobs.json` records the replacement ViT-S/B × 1in1/4in1 runs, all from scratch for 100 epochs. Each epoch selects 80,000 anchors without replacement from the full contact pool, changing deterministically with seed42+epoch; all four variants use identical sampled identities. Four ranks each receive 20,000 anchors, giving 1,250 global-batch64 updates per epoch. Val/test and best-epoch full-architecture saving are unchanged. This sampling change reduces work per epoch; it does not change the per-clip conversion or install the experimental mmap pipeline.

Current GR00T and PI adapters explicitly disable VLM gradient checkpointing. Earlier speed measurements in artifacts used checkpointing enabled and do not describe this updated setting.

## Precomputed radial cache and precision

`python -m training.cache_tactile --workers 16` precomputes every sensor frame from all contact episodes into `data/contact_depth_deform_radial_v1/<object>/<episode>/{left,right}.npy`. The cache is lossless FP32 NCHW; it stores each original frame once, with representation metadata, source size/mtime and streaming payload SHA256. Completed episode metadata enables safe resumption. Buffered sequential writes avoid slow NFS mmap write faults. `manifest.json` is published only after every episode succeeds.

The frozen 160 clips / 640 frames are also cached at `data/encoder_eval_v1/depth_deform_radial.npy`, with unchanged val/test identities and source raw-archive checksum in `depth_deform_radial.json`. Raw marker RGB, depth and marker coordinates remain intact. Cached readers assemble chronological stride-2 histories without CPU interpolation. `training.pretrain_rae` defaults to `--data-mode cached`; `--data-mode online` retains the reference path. `--cache-root` overrides the contact cache. Data/cache storage dtype is independent of training arithmetic precision.

The prepared training defaults are single GPU, micro/global batch64, 80,000 freshly sampled anchors per epoch and 100 epochs. No new DLC job is submitted by cache creation or precision testing.

Precision modes: `amp_bf16` (default) retains FP32 model parameters/AdamW moments and uses BF16 autocast; `bf16` also casts model parameters and native AdamW moments to BF16; `fp32` disables autocast. Loss reduction uses FP32 in every mode. Native BF16 speeds up the tested short runs by roughly 11–13% over AMP BF16 on A100 batch64, but changes optimizer numerical precision and is not a convergence-equivalent replacement. Measurements are recorded in `artifacts/rae_step_benchmark/precision.json`; no 100-epoch convergence comparison has been run. Checkpoint architecture metadata includes parameter dtype, and evaluation restores its training precision.

Both StarVLA adapters default to eight independent training noise/time samples per demonstration (`flow_repeats=8`). Inference draws one initial noise trajectory per demonstration and performs ten Euler updates; it does not repeat or ensemble the training noise samples. VLM gradient checkpointing and KV caching are disabled.

## RAE early stopping

The default training patience is 10 completed epochs without a strictly lower validation MSE. Each improvement resets the counter; equality counts as no improvement. Best full-model artifacts remain unchanged on non-improving epochs. Future launches stop in the trainer and evaluate the best checkpoint once before exiting.

The already-running immutable jobs in `runs/rae_cached_single64_100epoch_20261116` are protected by `training.early_stop_rae`: a detached local monitor checks completed-epoch metadata every 30 seconds, then stops only the corresponding DLC job after ten stale epochs. `status.json` is published after checkpoint saves, so the complete best/last checkpoints are retained. The current process/criterion/status are saved in `early_stopping.json`, with actions in `early_stopping.log`. These live jobs are not interrupted or resubmitted to install the rule.

## StarVLA 分阶段实验计划

**运行分工：策略与 RAE 训练均使用 DLC；仿真测试使用 ECS `ecs-5090-fifa`。两端通过共享 NAS 读取代码和 checkpoint。**

目标：比较无触觉、触觉进入 VLM、触觉进入 head。主指标仅为 8 个任务成功率的等权平均，不按速度或显存选优；并列时保留并列候选，追加相同数量、相同新增 seeds 的 rollout 再比较。

完整 VLM 网格为 `2×3×3×2×2×2×3=432` 组；144 组对应尚未乘入三种 attention mask。采用以下逐阶段筛选，暂缓 CLS，复用配置完全相同的已完成对照：

| 阶段 | 实验设置 | 新策略 checkpoint 数 |
| --- | --- | ---: |
| 历史无触觉基线 | 原 GR00T、PI 固定学习率结果保留作历史对照 | 2 |
| 新无触觉基线 | groot_v1、PI 从 Qwen 基础权重重新训练，统一 warmup＋余弦学习率；作为独立基线比较，不阻塞后续筛选 | 2 |
| RAE 预训练 | 1in1/4in1 × ViT-S/B，沿用既有任务；各自使用固定验证集 MSE 最优的 best.pt | 另计 4 个 encoder |
| VLM A | GR00T × none/vision/tactile 包裹 × none/1d/3d RoPE；固定 RAE-1in1-S、pool3x3、不冻结、causal | 9 |
| VLM B | 将 A 最优包裹/RoPE 放到 groot_v1 上；pool3x3/learned3x3 × 冻结/不冻结 RAE；新学习率 | 4（模型/学习率已变化，不能复用 A 的对照） |
| VLM C | 沿用 groot_v1 与新学习率，最优压缩/冻结方式；ViT-S/B × RAE-1in1/4in1 | 3（4 组中复用 1 组） |
| VLM D | 沿用 groot_v1 与新学习率，最优编码器；causal / 模态内双向且模态间 causal / 全双向 | 2（3 组中复用 causal） |
| VLM E | 将 groot_v1 的 D 最优设置迁移到 PI，使用新学习率 | 1 |
| Head | 沿用选出的编码器、压缩和冻结方式，groot_v1/PI 各 1 组，均使用新学习率 | 2 |

修订后合计 **25 个策略 checkpoint（含 2 个历史无触觉基线及 2 个新基线）+ 4 个 RAE encoder**，首轮共 `25×8×50=10000` 个有效 rollout。VLM A 的原 9 组继续跑完，不修改其架构或学习率；B 仅等待 A 完成评测并选出最优配置，不等待两项新无触觉基线；B/C/D 固定使用 groot_v1，随后 E 迁移到 PI。这里的复用只指完全一致的实验；每个新增策略均从同一 Qwen 基础权重及对应固定 RAE best.pt 独立训练，不从前一阶段策略继续训练。分阶段搜索不保证全局最优；若最终设置与第一阶段结论明显交互，再针对性复核前两名，额外实验另计。

无触觉基线可与 RAE 预训练并行。触觉下游等待对应 RAE 完成或早停，再把 best.pt 复制为不可变版本并记录 SHA256；不直接依赖仍可能被覆盖的在训 best.pt。

各阶段的 50 rollout/任务用于选优。最终对 6 组主结果（新学习率的 groot_v1/PI 无触觉各 1 组、VLM 各 1 组、head 各 1 组）使用独立 seeds 再测各 50 rollout/任务，另计 2400 次，避免将选优集结果当作独立测试结果。首次验证 seeds 从 1000000 开始；独立最终测试可设 START_SEED=2000000。候选使用相同 seed 序列；仿真或通信异常不算有效回合，记录异常 seed 并继续直到每任务 50 个有效回合。

固定条件：50,000 次 optimizer 更新、global batch=64、每卡 micro batch=16（不足 4 卡时累积补足）、训练 seed=42、相同 800 个 policy episode 与动作统计；7 维相对机械臂动作+绝对夹爪，chunk=50、执行前 30 步。新增训练统一学习率峰值 2.5e-5（VLM、head 及可训练触觉参数相同），warmup 1000 步，按 UniVTAC1 的余弦调度在第 30000 步降至 2.5e-6，后续保持；其他 AdamW 参数保持当前仓库设置。scheduler 状态随 checkpoint 保存并在续训时恢复。Action head 宽度 1024；训练每示范 8 组噪声，推理 1 份初始噪声、10 步去噪；关闭 VLM 梯度检查点。共享触觉输入为 depth_deform_radial、TPS28+bilinear，时序 stride=2；预训练与策略使用相同表示和帧语义。四组 RAE 在训任务及其 best/last 权重不清理、不重启。清理旧策略时只处理确认属于待重训 GR00T/PI 的策略 checkpoint，保留 encoder、统计、日志和数据。

代码顺序已确认（不计 chat 控制 tokens）：`tactile_fusion=vlm` 在最后一个视觉块之后、语言之前插入 T，即 **V→T→L**；GR00T 用最后层完整 VTL 条件，PI 用对应层完整 VTL 条件再拼 action KV。`tactile_fusion=head` 在 VLM 的 VL 输出之后追加 T，即 **V→L→T**，PI 再追加 action KV。Head 组继承编码器/压缩/冻结配置，T 的 VLM 包裹与 RoPE 不适用；V/L 的 Qwen mask 沿用选优设置。三种 Qwen mask 只改变 6 个标准 attention 层，18 个 GatedDeltaNet 保持原因果实现。

`learned3x3`：策略侧 pool3x3 基线加可学习 CNN 残差。CNN 为深度卷积 3×3/stride=2 → GELU → 自适应 3×3 → 1×1 通道混合。深度卷积以二项式低通核初始化，最后 1×1 权重置零，因此初始输出与 pool3x3 完全一致。CNN 在左右传感器和输出 latent 帧之间共享。冻结只作用于 RAE，包括时序压缩；CNN 和到 VLM/head 的投影仍训练。未改动在训 RAE 的架构或权重格式。

评测在 **ecs-5090-fifa** 执行，使用共享 NAS；SSH 到位后检查 Python/Isaac/CUDA 环境、实际代码和依赖、GPU 数量与 POLICY_GPU/SIM_GPU 分配，再进行短回合冒烟及正式 rollout。执行状态：整套计划已启动；实时状态以 `runs/starvla_plan_20261116/state.json` 为准。动作步数上限按 UniVTAC1：

| task | 最大执行动作步数 |
| --- | ---: |
| grasp_classify | 300 |
| insert_HDMI | 600 |
| insert_hole | 300 |
| insert_tube | 300 |
| lift_bottle | 500 |
| lift_can | 300 |
| pull_out_key | 300 |
| put_bottle_in_shelf | 600 |

这些是 take_action 计数，不是物理仿真步数或 chunk 次数。reset_time_limit=1200 秒，和动作上限分别控制。`training/eval_settings.json` 为评测设置，`training/evaluate.sh` 默认每任务 50 有效回合并显式传入各任务上限。

训练/推理一致性：固定 checkpoint 的 config、stats、encoder 来源及哈希、源码快照；严格加载策略权重（含 learned3x3），不在评测时换 encoder 或独立覆盖 token/freeze/包裹/RoPE/mask。先检查单帧/四帧历史、stride 与首帧 padding、左右传感器顺序、离线与在线 depth_deform 转换、动作还原及 chunk 执行语义，再验证同一输入经保存/加载后输出一致。报告每任务成功数/有效回合数及宏平均；中断或不足 50 回合标为未完成。

启动参数示例：`--tactile --encoder rae --encoder-checkpoint <best.pt> --rae-variant rae-1in1 --rae-size S --tokens learned3x3 --freeze-tactile-encoder --tactile-fusion vlm`。不冻结用 `--no-freeze-tactile-encoder`；head 组用 `--tactile-fusion head`。训练、DLC 打包及 smoke 入口均支持这些参数。

## 执行与等待

训练资源优先使用 DLC A800（`quota3ya3nugiuc6`）；A800 无可调度配额而 A100（`quota4wxjihmxlih`）有空闲时使用 A100。控制器按 GPU、CPU、内存及排队请求检查配额；仅迁移未启动的排队任务，确认旧任务停止后再提交，运行中的训练保持原资源池。

持久化控制器每 60 秒检查 DLC、RAE 最佳权重及 ECS 评测队列；最多保留 11 个已提交策略训练（包括排队任务），支持 2 个无触觉基线与 VLM A 的 9 组同时提交；实际训练并发由 DLC 资源调度决定。评测按任务拆分，最多 8 个评测进程并行。ECS 从 GPU 0–7 中选择显存低于 512 MiB 且未被本队列占用的卡，并使用 GPU 锁。一个评测在同卡运行模型服务与仿真；任务间复用模型服务。

训练完成需同时满足 DLC Succeeded、50,000 步日志及最终 checkpoint 存在。每回合结果原子保存，异常不计入 50 个有效回合；失败评测最多自动重试一次，提交状态不明时先核对唯一 DLC 名称，不重复提交。阶段并列最多追加三轮同量评测，仍并列则标记 NeedsAttention，不按成本破平局。

状态与结果目录：`runs/starvla_plan_20261116/`，包括 `state.json`、`controller.log`、`events.jsonl`、`results.csv`、`final_results.json`、冻结源码及 RAE 来源哈希。控制器进程信息见 `controller.json`。若调度机器停机，恢复后运行 `python -u -m training.experiment_plan --experiment runs/starvla_plan_20261116 --interval 60`；文件锁防止双重调度，已提交训练不会重新提交。ECS 完整链路先通过独立临时权重的短回合冒烟，之后才放行正式评测。邮件通知已取消。


2026-11-16 紧急暂停：本实验 A800 的 8 个作业已停止，控制器状态为 `Paused`，A100 作业保持运行。恢复清单见 `runs/starvla_plan_20261116/pause_a800.json`；逐作业的 `resume_submission.template.json` 仅为模板，不会自动提交。恢复时使用 `resume_source_v1` 和 `--resume <checkpoint>` 加载模型、AdamW、步数与 RNG，从保存的步数训练至总计 50,000 步。提交前生成唯一作业名、归档旧作业 ID 与日志，并将 train.jsonl 中超过保存步数的记录归档后截断；用户明确通知后才恢复控制器。原 checkpoint 每 5,000 步保存，未保存进度需重跑；旧文件未保存各 rank 的 Python/NumPy/数据 worker RNG，因此不能保证逐位一致续训。

2026-11-17：用户已通知恢复。8 个 A800 暂停任务均按原 checkpoint 使用 `--resume` 重新提交，继续到总计 50,000 步；控制器已恢复 Running，保持 A800 优先、A100 回退。恢复前日志保存在各任务 `paused_logs_<旧 job_id>/`，新旧作业关联与恢复步数见 state.json/job_history。保留运行的 3 个 A100 任务均已完成 50,000 步，进入 ECS 评测队列。

2026-11-17 评测调度：优先无触觉 PI，8 张 ECS 5090 各负责一个任务（每任务 50 有效 rollout，共 400）；保留已有回合并续测。后续 checkpoint 自动按任务分片调度，每个分片使用独立目录、模型服务端口及 GPU 锁，8 个任务完成后汇总平均成功率。

`starvla_groot_v1` 的 CA：Q=[1 个 state token, 50 个 action tokens]，KV=最后层完整 VLM 条件；无 future tokens。SA 保留 state+action 双向注意力。PI 的原 head 结构保持不变。新增源码冻结为 `frozen_source_v2`，历史 `frozen_source`、在训任务及其源码/权重保留。

2026-11-21 补测修复：eval_policy 的正常/异常出口统一显式关闭 task、policy、simulation_app，清理最多 20 秒后按原状态退出，避免 Isaac/Kit 析构卡住；attachment 复位异常记录为 error 后退出仿真实例，不在损坏状态连续重置。保留 1200 秒 reset 上限、原物理配置及成功判定。PI B pool3x3＋冻结的 insert_tube 已安排人工补测，原错误记录和其他 7 个任务保留；临时为它让卡的 lift_can 分片在本次补测结束后恢复。该退出修复不等于已验证 attachment 稳定性根因，补测以实际有效回合数为准。

2026-11-21：用户要求重新测试 PI pool3x3＋冻结 encoder 的 40k、50k checkpoint。两项均另建 retest 记录与独立目录，不复用旧 rollout，也不覆盖原 50k 的 76.50% 结果。每项 8 task × 50 有效回合，沿用起始 seed=1000000、训练配置及各任务动作步数上限。该补充共 800 有效 rollout，不混入阶段选优或最终独立测试。

2026-11-22 PI C 扩展：Pool3x3 × 冻结/不冻结 × ViT-S/B × 1in1/4in1，共 8 个配置；S-1in1 的两个已有对照复用，新增 S-4in1、B-1in1、B-4in1 各冻结/不冻结，共提交 6 项，每项 4 GPU、50k steps，训练后自动评测 8 task × 50 有效回合。A800 优先，A100 回退。原 40k/50k retest 保留为独立结果。
