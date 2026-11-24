# StarVLA 实验计划

**运行分工：策略与 RAE 训练均使用 DLC；仿真测试使用 ECS `ecs-5090-fifa`。两端通过共享 NAS 读取代码和 checkpoint。**

目标：比较无触觉、触觉进入 VLM、触觉进入 head。VLM A 按用户指定，两模型后续统一默认使用 **tactile 包裹＋1d RoPE**；A 网格结果保留作对照，不再根据其最高成功率更换默认配置。其余需选优阶段的主指标仅为 8 个任务成功率的等权平均，不按速度或显存选优；并列时保留并列候选，追加相同数量、相同新增 seeds 的 rollout 再比较。

## 实验矩阵与分阶段筛选

完整 VLM 网格为 `2×3×3×2×2×2×3=432` 组，其中两种模型为 **groot_v1、PI**；144 组对应尚未乘入三种 attention mask。groot_v1、PI 均执行 A→B→C→D；两路线分别选优、并行推进，暂缓 CLS，复用配置完全相同的已完成或在训对照：

| 阶段 | 实验设置 | 策略 checkpoint 数（扣除阶段间复用） |
| --- | --- | ---: |
| 历史无触觉基线 | 原 GR00T、PI 固定学习率结果保留作历史对照 | 2 |
| 新无触觉基线 | groot_v1、PI 从 Qwen 基础权重重新训练，统一 warmup＋余弦学习率；作为独立基线比较，不阻塞后续筛选 | 2 |
| RAE 预训练 | 1in1/4in1 × ViT-S/B，沿用既有任务；各自使用固定验证集 MSE 最优的 best.pt，两模型共用 | 另计 4 个 encoder |
| 历史 VLM A | 原 GR00T × none/vision/tactile 包裹 × none/1d/3d RoPE，沿用原架构和固定学习率；保留已有 9 组结果，不代替新 A 的模型内筛选 | 9 |
| VLM A | groot_v1、PI 各自比较 none/vision/tactile 包裹 × none/1d/3d RoPE；固定 RAE-1in1-S、pool3x3、不冻结、causal，使用新学习率；两模型默认选用 tactile＋1d，其余候选保留作对照 | 18（每模型 9 组；已有同配置任务复用） |
| VLM B | 每模型固定使用 A 指定的 tactile 包裹＋1d RoPE；各比较 pool3x3/learned3x3 × 冻结/不冻结 RAE，保持 RAE-1in1-S、causal 和新学习率 | 6（每模型 4 组中复用自身 A 的 1 组，新增 3 组） |
| VLM C | groot_v1 沿用 B 最优 pool＋冻结；PI 固定 pool3x3，同时比较冻结/不冻结；两模型均比较 ViT-S/B × RAE-1in1/4in1，保持 tactile＋1d、causal 和新学习率 | 9（groot_v1 新增 3；PI 新增 6，复用自身 S-1in1 冻结及不冻结对照） |
| VLM D | 每模型沿用自身 C 最优编码器及前序最优设置；causal / 模态内双向且模态间 causal / 全双向，均使用新学习率 | 4（每模型 3 组中复用自身 C 的 causal，新增 2 组） |
| Head | groot_v1、PI 各 1 组，分别继承自身路线选出的编码器、压缩、冻结及 Qwen mask，均使用新学习率 | 2 |

标准计划合计 **52 个策略 checkpoint（含 2 个历史无触觉基线、2 个新基线及 9 个历史 GR00T VLM A）+ 4 个 RAE encoder**，首轮共 `52×8×50=20800` 个有效 rollout；加上独立最终测试 2400 次，共 **23200** 次，阶段并列追加评测和额外复核另计。52 为完整计划的唯一配置总数，并非本次还需新提交的数量；已有且完全相同的配置直接复用，不重复训练或评测。原 VLM E 的“将 groot_v1 最优设置迁移到 PI”取消，由 PI 自己的 A/B/C/D 筛选替代。

两路线分别维护候选与 winner，不能将两个模型混在一起只选一个 winner。groot_v1、PI 的 A 均按用户指定选用 tactile＋1d，历史 GR00T A winner 及新 A 网格最高分保留作对照，不替换指定配置。两路线的 C 只等待自身 B，D 只等待自身 C，各模型 B 复用自身 A 的指定配置，不等待其他 A 对照完成；两路线互不等待，也不等待新无触觉基线。每个新增策略均从同一 Qwen 基础权重及对应固定 RAE best.pt 独立训练，不从前一阶段策略继续训练。

已提交的 groot_v1 B 四组均为 tactile＋1d，全部继续计入正式 B；其中 pool3x3＋不冻结的一组与指定 groot_v1 A 配置完全一致，直接复用。分阶段搜索不保证全局最优；若最终设置与第一阶段结论明显交互，再针对性复核前两名，额外实验另计。

无触觉基线可与 RAE 预训练并行。触觉下游等待对应 RAE 完成或早停，再把 best.pt 复制为不可变版本并记录 SHA256；不直接依赖仍可能被覆盖的在训 best.pt。

## 选优与最终测试

各阶段的 50 rollout/任务用于选优。最终对 6 组主结果（新学习率的 groot_v1/PI 无触觉各 1 组、VLM 各 1 组、head 各 1 组）使用独立 seeds 再测各 50 rollout/任务，另计 2400 次，避免将选优集结果当作独立测试结果。首次验证 seeds 从 1000000 开始；独立最终测试可设 START_SEED=2000000。候选使用相同 seed 序列；仿真或通信异常不算有效回合，记录异常 seed 并继续直到每任务 50 个有效回合。

## 统一训练条件

固定条件：50,000 次 optimizer 更新、global batch=64、每卡 micro batch=16（不足 4 卡时累积补足）、训练 seed=42、相同 800 个 policy episode 与动作统计；7 维相对机械臂动作+绝对夹爪，chunk=50、执行前 30 步。新增训练统一学习率峰值 2.5e-5（VLM、head 及可训练触觉参数相同），warmup 1000 步，按 UniVTAC1 的余弦调度在第 30000 步降至 2.5e-6，后续保持；其他 AdamW 参数保持当前仓库设置。scheduler 状态随 checkpoint 保存并在续训时恢复。Action head 宽度 1024；训练每示范 8 组噪声，推理 1 份初始噪声、10 步去噪；关闭 VLM 梯度检查点。共享触觉输入为 depth_deform_radial、TPS28+bilinear，时序 stride=2；预训练与策略使用相同表示和帧语义。四组 RAE 在训任务及其 best/last 权重不清理、不重启。清理旧策略时只处理确认属于待重训 GR00T/PI 的策略 checkpoint，保留 encoder、统计、日志和数据。

## 触觉融合与 learned3x3

代码顺序已确认（不计 chat 控制 tokens）：`tactile_fusion=vlm` 在最后一个视觉块之后、语言之前插入 T，即 **V→T→L**；GR00T 用最后层完整 VTL 条件，PI 用对应层完整 VTL 条件再拼 action KV。`tactile_fusion=head` 在 VLM 的 VL 输出之后追加 T，即 **V→L→T**，PI 再追加 action KV。Head 组分别继承同模型 A/B/C/D 路线的编码器/压缩/冻结配置，T 的 VLM 包裹与 RoPE 不适用；V/L 的 Qwen mask 沿用同模型选优设置。三种 Qwen mask 只改变 6 个标准 attention 层，18 个 GatedDeltaNet 保持原因果实现。

`learned3x3`：策略侧 pool3x3 基线加可学习 CNN 残差。CNN 为深度卷积 3×3/stride=2 → GELU → 自适应 3×3 → 1×1 通道混合。深度卷积以二项式低通核初始化，最后 1×1 权重置零，因此初始输出与 pool3x3 完全一致。CNN 在左右传感器和输出 latent 帧之间共享。冻结只作用于 RAE，包括时序压缩；CNN 和到 VLM/head 的投影仍训练。未改动在训 RAE 的架构或权重格式。

## 仿真评测设置

评测在 **ecs-5090-fifa** 执行，使用共享 NAS；SSH 到位后检查 Python/Isaac/CUDA 环境、实际代码和依赖、GPU 数量与 POLICY_GPU/SIM_GPU 分配，再进行短回合冒烟及正式 rollout。执行状态：整套计划已启动；旧 GR00T 策略 checkpoint 已清理，两项无触觉训练已提交 DLC。实时状态以 `runs/starvla_plan_20261116/state.json` 为准。动作步数上限按 UniVTAC1：

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

## 训练与推理一致性

训练/推理一致性：固定 checkpoint 的 config、stats、encoder 来源及哈希、源码快照；严格加载策略权重（含 learned3x3），不在评测时换 encoder 或独立覆盖 token/freeze/包裹/RoPE/mask。先检查单帧/四帧历史、stride 与首帧 padding、左右传感器顺序、离线与在线 depth_deform 转换、动作还原及 chunk 执行语义，再验证同一输入经保存/加载后输出一致。报告每任务成功数/有效回合数及宏平均；中断或不足 50 回合标为未完成。

## 启动参数

启动参数示例：`--tactile --encoder rae --encoder-checkpoint <best.pt> --rae-variant rae-1in1 --rae-size S --tokens learned3x3 --freeze-tactile-encoder --tactile-fusion vlm`。不冻结用 `--no-freeze-tactile-encoder`；head 组用 `--tactile-fusion head`。训练、DLC 打包及 smoke 入口均支持这些参数。

## 执行与等待

训练资源优先使用 DLC A800（`quota3ya3nugiuc6`）；A800 无可调度配额而 A100（`quota4wxjihmxlih`）有空闲时使用 A100。控制器按 GPU、CPU、内存及排队请求检查配额；仅迁移未启动的排队任务，确认旧任务停止后再提交，运行中的训练保持原资源池。

持久化控制器每 60 秒检查 DLC、RAE 最佳权重及 ECS 评测队列；最多保留 30 个已提交策略训练（包括排队任务）；groot_v1 A/B 与 PI A 在同一提交上限下并行推进，新候选全部提交 DLC 排队，不要求全部同时占用资源；实际训练并发由 DLC 资源调度决定。评测按任务拆分，最多 8 个评测进程并行。ECS 从 GPU 0–7 中选择显存低于 512 MiB 且未被本队列占用的卡，并使用 GPU 锁。一个评测在同卡运行模型服务与仿真；任务间复用模型服务。

训练完成需同时满足 DLC Succeeded、50,000 步日志及最终 checkpoint 存在。每回合结果原子保存，异常不计入 50 个有效回合；失败评测最多自动重试一次，提交状态不明时先核对唯一 DLC 名称，不重复提交。阶段并列最多追加三轮同量评测，仍并列则标记 NeedsAttention，不按成本破平局。

状态与结果目录：`runs/starvla_plan_20261116/`，包括 `state.json`、`controller.log`、`events.jsonl`、`results.csv`、`final_results.json`、冻结源码及 RAE 来源哈希。控制器进程信息见 `controller.json`。若调度机器停机，恢复后运行 `python -u -m training.experiment_plan --experiment runs/starvla_plan_20261116 --interval 60`；文件锁防止双重调度，已提交训练不会重新提交。ECS 完整链路先通过独立临时权重的短回合冒烟，之后才放行正式评测。邮件通知已取消。


2026-11-16 紧急暂停：本实验 A800 的 8 个作业已停止，控制器状态为 `Paused`，A100 作业保持运行。恢复清单见 `runs/starvla_plan_20261116/pause_a800.json`；逐作业的 `resume_submission.template.json` 仅为模板，不会自动提交。恢复时使用 `resume_source_v1` 和 `--resume <checkpoint>` 加载模型、AdamW、步数与 RNG，从保存的步数训练至总计 50,000 步。提交前生成唯一作业名、归档旧作业 ID 与日志，并将 train.jsonl 中超过保存步数的记录归档后截断；用户明确通知后才恢复控制器。原 checkpoint 每 5,000 步保存，未保存进度需重跑；旧文件未保存各 rank 的 Python/NumPy/数据 worker RNG，因此不能保证逐位一致续训。

2026-11-17：用户已通知恢复。8 个 A800 暂停任务均按原 checkpoint 使用 `--resume` 重新提交，继续到总计 50,000 步；控制器已恢复 Running，保持 A800 优先、A100 回退。恢复前日志保存在各任务 `paused_logs_<旧 job_id>/`，新旧作业关联与恢复步数见 state.json/job_history。保留运行的 3 个 A100 任务均已完成 50,000 步，进入 ECS 评测队列。

2026-11-17 评测调度：优先无触觉 PI，8 张 ECS 5090 各负责一个任务（每任务 50 有效 rollout，共 400）；保留已有回合并续测。后续 checkpoint 自动按任务分片调度，每个分片使用独立目录、模型服务端口及 GPU 锁，8 个任务完成后汇总平均成功率。

`starvla_groot_v1` 的 CA：Q=[1 个 state token, 50 个 action tokens]，KV=最后层完整 VLM 条件；无 future tokens。SA 保留 state+action 双向注意力。PI 的原 head 结构保持不变。新增源码冻结为 `frozen_source_v2`，历史 `frozen_source`、在训任务及其源码/权重保留。


2026-11-18 计划修订：VLM A/B/C/D 扩展为 groot_v1、PI 两条独立选优路线，均使用 warmup＋余弦学习率；历史 GR00T A 的 9 组保留。取消单独 VLM E 迁移阶段，最终每模型分别报告无触觉、VLM、Head 三组主结果。本文描述修订后的目标计划；已有 state.json 和控制器仍需同步为按模型区分的阶段及 winner 后，才会自动执行新增候选，不能把文档修订当作新增作业已提交。

2026-11-18 执行更新（取代上条中补跑 groot_v1 A 的安排）：用户指定 groot_v1 继续现有 VLM B；新增 PI VLM A 九组已接入控制器，阶段名为 `vlm_A_pi`，winner 独立记录。已保留原 `vlm_A` 历史结果和 `vlm_B` 在训任务。当前自动推进范围为现有 groot_v1 B 和新增 PI A，旧的共享 winner 下游推进已关闭；后续 C/D 及 PI B 仍需接入各自路线，不能按旧 VLM E 迁移逻辑生成。用户随后要求 PI A 九组直接全部提交 DLC 排队；提交上限已按用户最新要求调整为 30，容纳现有 4 个 groot_v1 B 和 9 个 PI A，实际并发由 DLC 调度。

2026-11-18 最新执行更新：用户随后授权 groot_v1 VLM A 一起提交，此条取代前述“不补跑 groot_v1 A”的安排。已启用 `vlm_A_pi`、`vlm_A_groot_v1` 两个独立阶段，分别记录 winner；groot_v1 A 九组中复用在训 `vlm_B_13_780901ac`（tactile＋1d、pool3x3、不冻结），新增八组。现有 groot_v1 B 四组继续运行。提交上限为 30，由 DLC 排队和分配实际训练并发。当前控制器自动创建与选优覆盖两模型 A，后续按各自 A winner 推进 B/C/D 仍待接入，旧共享 winner 自动推进保持关闭。

2026-11-20 默认配置更新：用户指定 VLM A 默认使用 tactile 包裹＋1d RoPE，两模型后续一致执行。控制器 `selection_overrides` 已记录指定配置：groot_v1 为 `vlm_B_13_780901ac`（77.50%），PI 为 `vlm_A_pi_24_a3e948d8`（80.25%）；PI A 按成功率选出的 vision＋none（80.75%）另存为 `measured_winners`，已有评测数据保持原样。指定配置不受其他 A 对照未完成的影响；后续 B/C/D 的压缩、冻结、编码器、mask 仍按同模型结果选优。

2026-11-20 PI VLM B 执行更新：已按用户指令启用 `vlm_B_pi`，固定 tactile 包裹＋1d RoPE、RAE-1in1-S、causal 和余弦学习率，比较 pool3x3/learned3x3 × 冻结/不冻结四组。pool3x3＋不冻结复用已完成的 `vlm_A_pi_24_a3e948d8`（80.25%），另外三组从基础权重独立训练并提交 DLC，训练完成后自动评测及独立选优；不从 A 策略续训。提交上限保持 30。此条取代此前 PI B 未接入的执行状态；C/D 和 Head 尚未启用自动推进。

2026-11-20 groot_v1 VLM C 执行更新：用户授权四组 C（复用一组），控制器已启用 `vlm_B_groot_v1`（完全复用现有四组 B）和 `vlm_C_groot_v1`。B 最优为 pool3x3＋冻结 RAE（78.50%），C 固定该设置及 tactile＋1d、causal、余弦学习率，对比 ViT-S/B × RAE-1in1/4in1。S＋1in1 复用 `vlm_B_14_5ced3256`，其他三组从基础权重独立训练并提交 DLC。四 encoder 的 SHA256 已重新核对；1in1 使用单帧、4in1 使用四帧且 stride=2。完成后自动评测并在 groot_v1 C 内选优；D、Head 尚未启用，PI C 尚未启用。

2026-11-21 补测修复：eval_policy 的正常/异常出口统一显式关闭 task、policy、simulation_app，清理最多 20 秒后按原状态退出，避免 Isaac/Kit 析构卡住；attachment 复位异常记录为 error 后退出仿真实例，不在损坏状态连续重置。保留 1200 秒 reset 上限、原物理配置及成功判定。PI B pool3x3＋冻结的 insert_tube 已安排人工补测，原错误记录和其他 7 个任务保留；临时为它让卡的 lift_can 分片在本次补测结束后恢复。该退出修复不等于已验证 attachment 稳定性根因，补测以实际有效回合数为准。

2026-11-21：用户要求重新测试 PI pool3x3＋冻结 encoder 的 40k、50k checkpoint。两项均另建 retest 记录与独立目录，不复用旧 rollout，也不覆盖原 50k 的 76.50% 结果。每项 8 task × 50 有效回合，沿用起始 seed=1000000、训练配置及各任务动作步数上限。该补充共 800 有效 rollout，不混入阶段选优或最终独立测试。

2026-11-22 PI C 扩展：Pool3x3 × 冻结/不冻结 × ViT-S/B × 1in1/4in1，共 8 个配置；S-1in1 的两个已有对照复用，新增 S-4in1、B-1in1、B-4in1 各冻结/不冻结，共提交 6 项，每项 4 GPU、50k steps，训练后自动评测 8 task × 50 有效回合。A800 优先，A100 回退。原 40k/50k retest 保留为独立结果。

2026-11-23 groot_v1 VLM D 执行更新：用户授权三种 Qwen attention mask 对比，控制器已启用 `vlm_D_groot_v1`。沿用 C 最优 ViT-B＋RAE-1in1、pool3x3、冻结 RAE、tactile 包裹＋1d RoPE及余弦学习率；causal 复用 `vlm_C_groot_v1_38_972f4ec2`（79.25%），modality（模态内双向、模态间 causal）和 bidirectional（全双向）各新增一组，从基础权重独立训练并提交 DLC。三种 mask 只改变 Qwen 的标准 attention 层，GatedDeltaNet 沿用原因果实现。训练完成后自动评测并独立选优。PI D、Head 和独立最终测试仍未启用。

2026-11-23 PI VLM D 执行更新：用户指定使用 ViT-B＋RAE-1in1、冻结 RAE，已启用 `vlm_D_pi`。固定 pool3x3、tactile 包裹＋1d RoPE、余弦学习率和其余训练条件；纯因果复用 `vlm_C_pi_44_be2948bf`（81.50%），新增 modality（模态内双向、模态间 causal）与 bidirectional（全双向）两组提交 DLC。仅标准 attention 层更换 mask，GatedDeltaNet 保持因果。两组均从 Qwen 基础权重和固化 encoder 独立训练至 50,000 步，完成后自动评测及 D 阶段选优，不从 C 策略续训。Head 与独立最终测试尚未启用。
