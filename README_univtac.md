<h1 align="center">UniVTAC</h1>

> UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking<br>
> [arXiv](https://arxiv.org/abs/2602.10093) | [PDF](https://arxiv.org/pdf/2602.10093) | [Website](https://univtac.github.io/) | [HuggingFace Dataset](https://huggingface.co/datasets/byml/UniVTAC) | [Modelscope Dataset](https://modelscope.cn/datasets/byml2024/UniVTAC)

**UniVTAC** is a tactile-aware simulation benchmark for robotic manipulation built on top of **NVIDIA Isaac Lab** and **TacEx (UIPC-based tactile simulation)**. It provides a unified framework for collecting expert demonstrations, training visuotactile policies, and evaluating them across a diverse suite of contact-rich manipulation tasks — all with high-fidelity tactile feedback from simulated GelSight Mini, ViTai GF225, or XenseWS sensors.

## News

- **2026-09** — UniVTAC was accepted to **CoRL 2026**.
- **2026-09** — Uploaded and updated the Isaac Sim 5.1 task dataset.

> [!IMPORTANT]
> The `isaac51` branch targets Isaac Sim 5.1, while the `main` branch preserves
> support for Isaac Sim 4.5. The Isaac Sim 5.1 release delivers up to **5x
> higher data-collection throughput** than the Isaac Sim 4.5 version and
> supports NVIDIA RTX 40- and 50-series GPUs. Data collected with the two
> versions is not cross-compatible. Both Isaac Sim 4.5 and 5.1 task datasets
> are available in the dataset repository; select the version that matches the
> code branch and simulator environment you use.

## Installation

The `main` branch targets Isaac Sim 5.1.0 and Isaac Lab 2.3.0. The installer creates or updates a Conda environment named `UniVTAC` with Python 3.11 and CUDA 12.6, then installs the modified local TacEx, libuipc and cuRobo dependencies.

```bash
git clone https://github.com/univtac/UniVTAC.git
cd UniVTAC
bash scripts/install.sh
```

See the [Installation Guide](./resources/docs/Installation.md) for CUDA/compiler prerequisites and verification, and the [migration notes](./resources/docs/isaacsim_5_1_migration.md) for the new GelSight, Actor and render-pipeline behavior.

## Task Gallery

UniVTAC currently includes the following manipulation tasks, all featuring tactile sensing:

| Task | Module | Description |
|---|---|---|
| **Collect** | `collect` | Collect contact-rich tactile data for pretraining |
| **Lift Bottle** | `lift_bottle` | Grasp and lift a bottle off a surface near a wall |
| **Lift Can** | `lift_can` | Grasp and lift a cylindrical can |
| **Insert HDMI** | `insert_HDMI` | Insert an HDMI connector into a port |
| **Insert Hole** | `insert_hole` | Precision peg-in-hole insertion |
| **Insert Tube** | `insert_tube` | Insert a tube into a fixture |
| **Pull Out Key** | `pull_out_key` | Extract a key from a lock |
| **Put Bottle in Shelf** | `put_bottle_in_shelf` | Place a bottle onto a shelf |
| **Grasp & Classify** | `grasp_classify` | Grasp an object and classify it by tactile feedback |

To build more tasks, refer to the [Task Creation Guide](./resources/docs/TaskCreation.md) for instructions on how to define new manipulation tasks within the UniVTAC framework.

## Data Collection

See the [Data Collection Guide](./resources/docs/Collection.md) for instructions on how to run the automated data collection pipeline, configure task-specific parameters, and understand the output data structure.

The dataset is available from [HuggingFace](https://huggingface.co/datasets/byml/UniVTAC)
and [ModelScope](https://modelscope.cn/datasets/byml2024/UniVTAC). Its published
layout is:

| Path | Contents |
|---|---|
| `isaac45/<task>/` | Isaac Sim 4.5 demonstrations: 100 HDF5 episodes and one `metadata.json` for each of the 8 tasks |
| `isaac51/<task>/` | Isaac Sim 5.1 demonstrations: 100 HDF5 episodes and one `metadata.json` for each of the 8 tasks |
| `contact/<shape>/` | Contact-pretraining trajectories for 14 non-empty shapes (638 HDF5 episodes in the current release) |
| `checkpoints/` | Policy checkpoints, metadata, dataset statistics, logs, and the shared tactile encoder; the current checkpoint release is trained for the Isaac Sim 4.5 dataset only |

Task trajectories are stored under `hdf5/*.hdf5`. Because simulator versions
are not interchangeable, downloading task data requires an explicit version:

```bash
# All Isaac Sim 5.1 task demonstrations
bash scripts/download_data.sh --task --version 51

# One Isaac Sim 4.5 task
bash scripts/download_data.sh --task lift_can --version 45

# Other independently selectable dataset components
bash scripts/download_data.sh --contact
bash scripts/download_data.sh --checkpoint

# Select multiple components in one invocation
bash scripts/download_data.sh --task --version 51 --contact --checkpoint
```

Selectors may be narrowed and repeated, for example
`--contact Cross --contact Sphere` or
`--checkpoint lift_can --checkpoint insert_hole`. Files are placed under
`resources/data/` by default while preserving the paths shown above. Run
`bash scripts/download_data.sh --help` for output-directory, revision,
parallelism, and force-download options.

> [!NOTE]
> The currently released policy checkpoints are associated with the Isaac Sim
> 4.5 demonstrations. They should not be treated as Isaac Sim 5.1 checkpoints.

## Train & Eval Policies

UniVTAC includes several baseline policies implemented under the `policy/` directory:

- ACT: Action Chunking with Transformers with/without tactile inputs
- Abation: ACT ablation variants for modality comparison
- ViTAL: ACT with CLIP-pretrained tactile-vision encoders in ViTAL
- PI0.5: PaliGemma prefix and Gemma action expert, under `policy/pi05/`
- StarVLA-GR00T: Qwen vision-language backbone and GR00T action head, under `policy/starvla_groot/`

PI0.5 and StarVLA-GR00T share infrastructure in `policy/vla_common/`. Unified training and checkpoint testing use the root `train_policy.sh` and `test_policy.sh`; see [policy training](policy/TRAINING.md).

Each policy is a self-contained module under `policy/` with its own data processing, training, and deployment scripts. All policies share a unified evaluation entry point at the project root:

```bash
bash test_policy.sh ${task_name} ${task_config} ${policy_config} ${gpu_id}
```

For parallel evaluation over many seeds:

```bash
bash scripts/parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
```

The evaluation results, including videos and success rate logs, will be saved in the `eval_result/` directory under the project root.

To deploy your own policy, refer to the [Deploy Your Policy](./resources/docs/Deploy.md).

## TODO

- Data collection and evaluation are now only supported on the GelSight Mini sensor. We will add support for ViTai GF225 and XenseWS in the near future.

## 👍 Citations
If you find our work useful, please consider citing:

```
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

## Contact
<div style="text-align: center;">
  <img src="https://box.nju.edu.cn/seafhttp/f/fc1021a908ff49309f22/?op=view" alt="Wechat Group" width="300"/>
</div>
