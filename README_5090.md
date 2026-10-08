# RTX 5090 / Isaac Sim 5.1 手动安装与验证

本文件记录在 Ubuntu 22.04、RTX 5090 上为 UniVTAC `isaac51` 分支建立**测试** Conda 环境的命令。2026-10-08 已在本机实际安装，并通过 PyTorch CUDA、`torch_scatter` CUDA、libuipc / cuRobo 导入及 Taxim、Pix2Pix 两种 Isaac Sim 无窗口触觉测试。按顺序逐段执行；每段通过检查后再进入下一段。不要执行 `scripts/install.sh`：它当前固定 CUDA 12.6、cu126 和 `sm_89`。训练用 `requirements_train.txt` 同样固定 cu126，不适合直接装入这个环境。

目标：Python 3.11；Isaac Sim 5.1.0；Isaac Lab 2.3.0；PyTorch 2.7.0 + cu128；CUDA Toolkit 12.8；5090 的 `sm_120`；仓库内 TacEx / libuipc，以及固定提交的 cuRobo。Conda 安装在 `/mnt/data_ssd/yichi.zhang/miniconda3`。以下命令从仓库根目录执行。

## 1. 前置检查

```bash
cd /mnt/data_ssd/yichi.zhang/UniVTAC
git branch --show-current                  # 应为 isaac51
nvidia-smi                                  # 应识别 RTX 5090，驱动 580.178.04
ldd --version                               # 第一行应为 GLIBC 2.35 或更新
df -h .
command -v git gcc g++ zip unzip pkg-config
```

若缺少系统构建工具，先由管理员安装 `build-essential curl git pkg-config unzip zip`。这一步不需要更换驱动。Isaac Sim 第一次运行时须由使用者阅读并接受 NVIDIA Omniverse EULA。

## 2. 创建环境和 CUDA 12.8 编译工具链

```bash
source /mnt/data_ssd/yichi.zhang/miniconda3/etc/profile.d/conda.sh
conda create -n univtac python=3.11 pip -y --override-channels -c conda-forge
conda activate univtac
conda install -y --override-channels -c nvidia -c conda-forge cuda-toolkit=12.8 cmake=3.26 ninja pkgconfig gcc_linux-64=12 gxx_linux-64=12 zip unzip libglu
python -m pip install --upgrade pip
python -m pip install setuptools==75.8.2 setuptools-scm==8.1.0 wheel==0.42.0 toml
python -m pip install flatdict==4.0.1 --no-build-isolation
python --version
"$CONDA_PREFIX/bin/nvcc" --version       # release 12.8
```

不要执行 `conda env update -f resources/third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml`；该文件固定 `cuda-toolkit=12.6`。

## 3. Isaac Sim、Isaac Lab 和 5090 PyTorch

Isaac Lab 2.3 的官方文档给出的 Linux x86_64 PyTorch 版本为 `2.7.0` 的 cu128 构建。先安装模拟器与 Lab，再最后固定 PyTorch，防止依赖解析换回其他 CUDA 构建。

```bash
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install 'isaaclab[isaacsim,all]==2.3.0' --extra-index-url https://pypi.nvidia.com
python -m pip install 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print((torch.ones(4, device="cuda") * 2).tolist())'
```

检查点：PyTorch 应为 `2.7.0+cu128`、`torch.version.cuda` 为 `12.8`、GPU capability 为 `(12, 0)`，CUDA 计算结果为 `[2.0, 2.0, 2.0, 2.0]`。`pip check` 如发现版本冲突，先定位具体包，记录解决方式；不要盲目执行 `pip install -U`。

## 4. 安装仓库内 TacEx 与 cu128 torch_scatter

仓库的 `resources/third_party/TacEx/source/tacex/setup.py` 写死了 cu126 `torch_scatter` 的 URL。先安装匹配 cu128 的预编译 wheel，再用 `--no-deps` 安装本地 TacEx，避免拉回 cu126。另将其省略的运行依赖显式安装。

```bash
python -m pip install --no-index --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html 'torch_scatter==2.1.2+pt27cu128'
python -m pip install debugpy psutil nvidia-ml-py pybind11 mypy transforms3d tetgen 'polyscope>=2.5,<3'
python -m pip install --no-build-isolation --no-deps -e resources/third_party/TacEx/source/tacex
python -m pip install --no-build-isolation -e resources/third_party/TacEx/source/tacex_assets
python -c 'import torch, torch_scatter; x=torch.tensor([1.,2.,3.],device="cuda"); i=torch.tensor([0,1,0],device="cuda"); print(torch_scatter.scatter_add(x,i,dim=0).tolist())'
```

检查点：GPU scatter 结果应为 `[4.0, 2.0]`。仅安装成功并不能证明 wheel 包含 5090 所需的 GPU 内核；此运算检查是必要的。`pip check` 对 TacEx 记录的 cu126 直接 URL 可能报告依赖不一致，此时以实际安装的 `torch_scatter` 版本与 GPU 运算结果为准，并记录该元数据问题；正式固定环境时应把 `setup.py` 的 URL 改成 cu128。

## 5. 配置编译变量并准备 vcpkg

这些变量必须在**编译 libuipc 与 cuRobo 的同一个 shell**中设置。重新打开终端后需重新激活环境并重设。`sm_120` 对应 `CMAKE_CUDA_ARCHITECTURES=120` 和 `TORCH_CUDA_ARCH_LIST=12.0`。

```bash
conda activate univtac
cd /mnt/data_ssd/yichi.zhang/UniVTAC
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST=12.0
export CMAKE_BUILD_PARALLEL_LEVEL=8
export MAX_JOBS=8
export UNIVTAC_GCC12="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export UNIVTAC_GXX12="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CC="$PWD/scripts/toolchains/gcc12-system-ld"
export CXX="$PWD/scripts/toolchains/gxx12-system-ld"
export CUDAHOSTCXX="$CXX"
export VCPKG_ROOT="$PWD/.cache/toolchains/vcpkg"
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
mkdir -p .cache/toolchains
git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
git -C "$VCPKG_ROOT" checkout --detach dd3097e305afa53f7b4312371f62058d2e665320
"$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
```

已有 vcpkg 目录时，先检查提交和可执行文件，不要重复克隆或覆盖已有工作。上述 GCC 包装脚本来自本仓库，使用系统链接器处理 Conda GCC 12 的链接问题。若工具链编译占用过多内存，把两个并行数调为 4。

## 6. 编译 libuipc 和固定版本 cuRobo

```bash
python -m pip install --no-build-isolation -e resources/third_party/TacEx/source/tacex_uipc
python -c 'import uipc; print("UIPC import OK")'

git clone https://github.com/NVlabs/curobo.git resources/third_party/curobo
git -C resources/third_party/curobo checkout --detach ebb71702f3f70e767f40fd8e050674af0288abe8
python -m pip install 'numpy<2' numpy-quaternion yourdfpy importlib_resources scikit-image
python -m pip install --no-build-isolation --no-deps -e resources/third_party/curobo
python -c 'import curobo.curobolib.geom_cu; print("cuRobo CUDA extension import OK")'
python -m pip check
```

本机编译 libuipc 时，vcpkg 的 `tinygltf` 2.9.6 下载包与登记的 SHA512 不一致。实际下载两次得到相同 SHA512 后，用以下命令建立本地 port 覆盖并重试。只有在重新下载并核实哈希后才使用这个本机变通方案；不要跳过完整性验证。

```bash
sha512sum "$VCPKG_ROOT/downloads/syoyo-tinygltf-v2.9.6.tar.gz"
# 本机结果：f736b30a55fcbb3b80bf25240e0ab2f50c57c380e1425efc30ce1b987f9167ea4b56d98f89e34947c3d77f1fb0b895c01e2dbfa10ddf7210c320ec55fd77b700
mkdir -p .cache/vcpkg-overlays/tinygltf
cp "$VCPKG_ROOT/ports/tinygltf/portfile.cmake" "$VCPKG_ROOT/ports/tinygltf/vcpkg.json" .cache/vcpkg-overlays/tinygltf/
sed -i 's/89397dc2c8884a54ea0c370251449459a200057b5e470210c4468f43c4623947500630b1a67ff6319e0998e648487367398f134711bc7d2c42ebdbd7097770b3/f736b30a55fcbb3b80bf25240e0ab2f50c57c380e1425efc30ce1b987f9167ea4b56d98f89e34947c3d77f1fb0b895c01e2dbfa10ddf7210c320ec55fd77b700/' .cache/vcpkg-overlays/tinygltf/portfile.cmake
export VCPKG_OVERLAY_PORTS="$PWD/.cache/vcpkg-overlays"
export CMAKE_GENERATOR=Ninja
python -m pip install --no-build-isolation -e resources/third_party/TacEx/source/tacex_uipc
```

libuipc 的 `setup.py` 使用固定的 `resources/third_party/TacEx/source/tacex_uipc/build` 目录。若其中已有其他 Python、CUDA 或 vcpkg 配置的 `CMakeCache.txt`，先检查并备份旧构建产物，再清理该**生成的构建目录**后重试。不要清理源码或尚未确认用途的目录。cuRobo 目录若已存在，也先检查其 Git 提交及本地修改。实际构建日志分别在 `.cache/uipc-build.log`、`.cache/curobo-build.log`；两者均使用 `sm_120`。

## 7. 仿真和项目验收

先使用一张空闲 GPU，运行 `num_envs=1` 的无窗口测试。首次运行可能下载扩展与在线资产。EULA 只在使用者已接受协议后，通过交互提示或环境变量处理。

```bash
conda activate univtac
cd /mnt/data_ssd/yichi.zhang/UniVTAC
unset LD_LIBRARY_PATH
export OMNI_KIT_ACCEPT_EULA=YES  # 仅限已阅读并接受 NVIDIA Omniverse EULA 的使用者
export LD_PRELOAD="$CONDA_PREFIX/lib/libGLU.so.1"
CUDA_VISIBLE_DEVICES=0 python -u scripts/smoke_isaac51.py --backend taxim --headless
CUDA_VISIBLE_DEVICES=0 python -u scripts/smoke_isaac51.py --backend pix2pix --headless
CUDA_VISIBLE_DEVICES=0 python -u scripts/collect_data.py grasp_classify demo --start_seed 0 --max_seed 0 --headless --config-overrides collect_settings.episode_num=1 collect_settings.use_seed=false collect_settings.save_root_dir=.cache/collect-smoke
```

编译时需要 Conda 的 `LD_LIBRARY_PATH`，但本机运行 Isaac Sim 时保留整个 `LD_LIBRARY_PATH` 会使 Kit 的 GPU 插件启动崩溃；运行时取消该变量，仅预加载 `libGLU.so.1`。`scripts/smoke_isaac51.py` 的任务模式已改成受机器人配置支持的 `eval`。Taxim 与 Pix2Pix 已各自输出 `PASS`：两只 GelSight Mini 输出 `240×320×3` 图像、UIPC 与相机更新、机械臂运动后标记跨度检查均通过。数据采集入口已在 seed 0 完成 1/1 回合，写出 59 帧和 `.cache/collect-smoke/grasp_classify/demo/hdf5/0.hdf5`。日志在 `.cache/smoke-taxim-unbuffered.log`、`.cache/smoke-pix2pix.log` 和 `.cache/collect-smoke.log`。环境清单在 `.cache/univtac-conda-explicit.txt`、`.cache/univtac-pip-freeze.txt`。Isaac Sim 的 RTX/PhysX 设备选择需结合日志核对，不能仅凭 `CUDA_VISIBLE_DEVICES` 判断。最后再加入所选策略的**推理**依赖、5.1 对应 checkpoint，运行 `test_policy.sh`。公开的旧 checkpoint 对应 Isaac Sim 4.5 数据，不能作为 5.1 策略成功率的验收材料。

通过后记录：`conda list --explicit`、`python -m pip freeze`、`nvidia-smi`、`nvcc --version`、实际 Git 提交、三项运行日志与所用 checkpoint。安装遇到错误时停在当前段修复，不继续叠加依赖。

## 8. StarVLA-GR00T checkpoint 推理测试

已测试 `policy_results/20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder/checkpoint_30000.pt`。它只包含 `lift_can` 任务，目录中的 `inference_assets/qwen` 有配置、processor 和 tokenizer，但没有独立的 Qwen 权重。部署代码现在优先读取该目录，按配置创建 Qwen，随后从完整的策略 checkpoint 恢复参数；无需另行下载 Qwen3.5-2B 权重。

在现有 `univtac` 环境中补齐推理依赖（保持 PyTorch 的 cu128 构建）：

```bash
conda activate univtac
cd /mnt/data_ssd/yichi.zhang/UniVTAC
python -m pip install 'accelerate==1.12.0' 'draccus==0.10.0' 'diffusers==0.35.2' 'timm==1.0.26' 'transformers==5.5.0' 'tokenizers==0.22.2' 'ninja==1.13.0' 'chardet==5.2.0'
python -m pip install --no-deps 'causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
python -m pip check
```

`causal-conv1d` 的 GPU 运算已通过。训练依赖中的 `fla-core==0.5.0` 和 `flash-linear-attention==0.5.0` 虽能安装，但在此机的 PyTorch 2.7 / Triton / 5090 (`sm_120`) 组合上运行 GPU 内核时触发 `computeCapability not supported`，因此测试环境未保留这两个包；Qwen 自动使用已通过测试的 PyTorch 实现。

```bash
export OMNI_KIT_ACCEPT_EULA=YES  # 仅限已阅读并接受协议的使用者
export LD_PRELOAD="$CONDA_PREFIX/lib/libGLU.so.1"
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 bash test_policy.sh --checkpoint policy_results/20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder --inspect
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash test_policy.sh --checkpoint policy_results/20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder --task lift_can --offline --hdf5 .cache/collect-smoke/grasp_classify/demo/hdf5/0.hdf5
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash test_policy.sh --checkpoint policy_results/20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder --task lift_can --task-config demo --total-num 1 --start_seed 0 --max_seed 0 --headless --config-overrides replay_settings.save_root_dir=.cache/policy-eval
```

离线命令使用前一节生成的 `grasp_classify` 数据，且没有本地 `lift_can` HDF5；因此它只检验加载和输出形状，不能用来判断动作质量。验证结果：离线推理成功加载第 30000 步，输出 `32×8` 动作序列；真实 `lift_can` 仿真在 seed 0 执行满 300 个策略动作，任务未成功（`0/1`），没有运行异常。这是单个种子的功能测试，不能视为策略成功率评估。日志为 `.cache/starvla-offline-final.log` 和 `.cache/starvla-sim.log`，仿真结果及视频在 `.cache/policy-eval/unified_deploy/lift_can/` 下。

## 参考

- [Isaac Sim 5.1 Python 环境安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Isaac Lab 2.3 pip 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
- [PyG PyTorch 2.7 + cu128 wheels](https://data.pyg.org/whl/torch-2.7.0+cu128.html)
- [RTX 5090 计算能力](https://developer.nvidia.com/cuda/gpus)
