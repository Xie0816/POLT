# POLT：基于本体感受的 UGV 通行性在线学习框架

## 项目概述

本仓库提供 **POLT** 的主体代码实现。

POLT 面向**复杂越野环境下的 UGV 通行性预测**问题。在这类场景中，地形的视觉外观与真实物理可通行性往往并不一致。POLT 的核心思想是：利用**本体感受反馈**进行**在线学习**，并通过**动态记忆机制**积累历史经验，从而预测车辆前方区域的地形通行代价。

当前仓库主要围绕论文中的主体实验流程组织，包括：

- 传感器数据处理
- 基于本体感受的反馈估计
- 在线 / 离线学习
- 特征点云融合
- 基于动态记忆的通行性代价预测

说明：仓库中保留 `planning/` 作为双代价 preset 的可选扩展；当前版本的重点仍然是**感知 + 在线学习**主流程。

## 功能特点

- 统一运行入口：`polt.py`
- 统一实验配置：`config.py`
- 支持 **5 个论文相关实验 preset**
- 支持 LiDAR、图像和本体感受数据处理
- 支持两类记忆机制：
  - `dynamic_memory`
  - `max_similiarity_out`
- 支持单反馈与双反馈通行性学习流程
- 提供日志记录与运行时 profiling 工具

## 仓库结构

```text
POLT/
├── polt.py
├── config.py
├── sensors/
│   ├── image.py
│   ├── lidar.py
│   └── proprio.py
├── polt_runtime/
│   ├── memory.py
│   ├── online_learning.py
│   ├── perception.py
│   ├── projection.py
│   ├── traversability.py
│   ├── visualization.py
│   └── runtime/
│       ├── coordinator.py
│       ├── data.py
│       ├── experiment_setup.py
│       ├── frame_pipeline.py
│       ├── profiling.py
│       └── analysis.py
├── model/
│   ├── vision/
│   ├── proprio/
│   ├── memory/
│   └── third_party/
├── planning/
│   ├── mppi.py
│   ├── planner.py
│   ├── types.py
│   └── visualization.py
├── env/
│   ├── environment.yml
│   ├── requirements-dinov3-cu118.txt
│   ├── setup_polt_env.sh
│   └── README.md
└── utils/
    ├── extract_dino_features.py
    ├── export_dino_features.py
    ├── export_dino_features_by_click.py
    ├── quick_add.py
    ├── train_vlad_database.py
    └── train_vlad_from_folder.py
```

### 代码功能映射

- `polt.py`：命令行入口
- `config.py`：preset 定义与运行配置
- `sensors/`：传感器级预处理
- `polt_runtime/online_learning.py`：主实验流程
- `polt_runtime/memory.py`：记忆机制 backend
- `polt_runtime/perception.py`：多传感器同步与特征帧构建
- `polt_runtime/traversability.py`：反馈估计与通行性统计
- `polt_runtime/runtime/`：运行时调度、数据读取、帧级辅助、profiling、分析
- `model/vision/`：DINOv3 推理封装与 VLAD 特征工具
- `model/proprio/`：本体感受力学模型与 SALON 粗糙度代价模型
- `model/memory/`：dynamic memory 与 data-buffer GPR 实现
- `model/third_party/`：随仓库保留的第三方模型源码，目前为 DINOv3
- `planning/`：可选双代价 risk map 构建、MPPI 后端与规划可视化
- `utils/`：离线 DINO 特征导出与 VLAD 训练工具

不属于主 runtime 模型接口的历史 DINO project 输出已移动到 `outputs/legacy_dino_project/`。

## 安装说明

推荐环境配置文件位于 `env/` 目录下，默认 conda 环境名为 `polt`。

代码默认面向 **Linux + NVIDIA GPU + CUDA 兼容驱动** 环境。当前测试环境使用 CUDA 11.8 PyTorch wheel 依赖栈。

### 方式一：脚本安装

在仓库根目录运行：

```bash
bash env/setup_polt_env.sh
conda activate polt
python polt.py --help
```

该脚本会：

- 创建 `polt` conda 环境；
- 安装 `env/requirements-dinov3-cu118.txt` 中固定的 Python 依赖；
- 安装本地 DINOv3 源码 `model/third_party/dinov3`；
- 执行一次基础 import 与版本检查。

### 方式二：手动安装

```bash
conda create -y -n polt python=3.11 pip
conda activate polt
python -m pip install --upgrade pip
python -m pip install -r env/requirements-dinov3-cu118.txt
python -m pip install -e model/third_party/dinov3
python polt.py --help
```

### 已测试核心版本

| 组件 | 版本 |
|---|---|
| Python | 3.11.13 |
| PyTorch | 2.7.1+cu118 |
| TorchVision | 0.22.1+cu118 |
| CUDA wheel runtime | 11.8 |
| torch-scatter | 2.1.2 |
| NumPy | 2.2.6 |
| OpenCV | 4.12.0 |
| Open3D | 0.19.0 |
| GPyTorch | 1.14.3 |
| Transformers | 4.57.1 |

更多环境细节见 `env/README.md`。

安装前建议确认 NVIDIA 驱动可用：

```bash
nvidia-smi
```

GitHub 仓库不包含模型权重、数据集和 vendored DINOv3 源码。请按下文准备
DINOv3 源码与权重，并根据本地路径放置数据集。

## DINOv3 源码与权重准备

POLT 使用 DINOv3 作为图像特征 backbone。当前代码在
`model/vision/dinov3_infer.py` 中按以下方式加载：

- 源码路径：`model/third_party/dinov3`
- 默认模型 key：`dinov3_vits16plus`
- 默认权重：
  `weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`

GitHub 仓库不直接包含 DINOv3 源码。运行 POLT 前，请从官方仓库下载：

```bash
git clone https://github.com/facebookresearch/dinov3.git model/third_party/dinov3
python -m pip install -e model/third_party/dinov3
```

DINOv3 权重文件较大，并且需要遵循原始 DINOv3 的 license 与访问流程，
因此不会提交到本仓库，环境脚本也不会自动下载。准备 POLT 默认权重时，
请按以下步骤操作：

1. 打开 DINOv3 官方下载页面：
   <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>
2. 申请访问权限并接受对应 license。
3. 使用 DINOv3 邮件中提供的下载 URL，下载
   **ViT-S+/16 distilled, LVD-1689M** 权重。
4. 将权重保存为 POLT 默认读取的文件名：

```bash
mkdir -p weights/dinov3
wget "<URL_FROM_DINOV3_ACCESS_EMAIL>" \
  -O weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

仓库中也可以保留可选的 ViT-S/16 权重：

```text
weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

DINOv3 官方模型也发布在 Hugging Face collection：
<https://huggingface.co/collections/facebook/dinov3>。但当前 POLT runtime
默认使用本地 `.pth` 文件，并通过 `torch.hub.load(..., source="local",
weights=...)` 加载；如果希望直接使用 Hugging Face 加载方式，需要同步修改
`model/vision/dinov3_infer.py`。

推荐目录结构：

```text
POLT/
├── model/
│   └── third_party/
│       └── dinov3/
└── weights/
    └── dinov3/
        ├── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
        └── dinov3_vits16_pretrain_lvd1689m-08c60483.pth  # 可选
```

## 数据集准备

`--root` 必须指向**数据集根目录**。

根据 `polt_runtime/runtime/data.py` 的读取逻辑，`root` 下必须包含以下内容：

- `image_front/`
- `lidar/`
- `lidarodometry.txt`
- `proprio_infos.json`

推荐组织方式如下：

```text
/path/to/dataset_root/
├── image_front/
│   ├── 1739873123456.jpg
│   ├── 1739873123556.jpg
│   └── ...
├── lidar/
│   ├── 1739873123456.bin
│   ├── 1739873123556.bin
│   └── ...
├── lidarodometry.txt
└── proprio_infos.json
```

说明：

- 图像与 LiDAR 文件名应使用**时间戳数字文件名**。
- `lidarodometry.txt` 以数值文本形式读取。
- `proprio_infos.json` 应为以时间戳为键的 JSON 字典。
- 请根据本地数据格式和路径进行调整（please adjust according to your local setup）。

注意：

- 当前 `polt.py` 中为 `--root` 保留了一个开发阶段的本地默认路径。
- 实际使用时，建议始终**显式传入 `--root`**。

## 快速开始

### 查看帮助

```bash
conda run -n polt python polt.py --help
```

### 运行 `online_learning`

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset online_learning
```

### 运行 `offline_learning`

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset offline_learning
```

### 运行 `online_learning_dual_cost`

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset online_learning_dual_cost \
  --mechanical-mem-buffer /path/to/mechanical_buffer \
  --roughness-mem-buffer /path/to/roughness_buffer
```

如果希望在 dual-cost 模式下关闭 planner 分支：

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset online_learning_dual_cost \
  --mechanical-mem-buffer /path/to/mechanical_buffer \
  --roughness-mem-buffer /path/to/roughness_buffer \
  --disable-planner
```

## 实验 Presets

当前代码在 `config.py` 中定义了如下 preset：

| Preset | 反馈类型 | 记忆机制 | 更新方式 | 默认日志文件 | 说明 |
|---|---|---|---|---|---|
| `offline_learning` | `力学` | `dynamic_memory` | `offline` | `log_offline.json` | 离线评估，不进行在线更新 |
| `online_learning` | `力学` | `dynamic_memory` | `online` | `log.json` | 单反馈在线学习主设置 |
| `online_learning_dual_cost` | `力学 + roughness` | `dynamic_memory` | `online` | `log_dual_cost.json` | 双反馈设置，默认启用 planner |
| `online_learning_with_databuffer` | `力学` | `max_similiarity_out` | `online` | `log_databuffer.json` | 缓冲区式记忆对比方法 |
| `salon` | `roughness` | `max_similiarity_out` | `online` | `log_salon.json` | 粗糙度反馈对比方法 |

### 主要配置轴

- `feedback_mode`
  - `mechanical`：力学反馈
  - `roughness`：粗糙度反馈
  - `both`：力学 + 粗糙度
- `memory_mode`
  - `dynamic_memory`
  - `max_similiarity_out`
- `update_mode`
  - `online`
  - `offline`

说明：

- 代码内部仍兼容旧别名 `fixed_size_data_buffer`
- 当前对外统一使用的名字是 `max_similiarity_out`

## 输出结果

默认情况下，不同 preset 会保存不同日志文件：

- `log.json`
- `log_offline.json`
- `log_databuffer.json`
- `log_salon.json`
- `log_dual_cost.json`

输出位置与模式有关：

- 对于不包含 planner 输出的 preset，日志通常保存在：
  - `<root>/<log_name>`
- 对于 `online_learning_dual_cost`，如果 planner 保持启用，日志可能保存在：
  - `planner_traces/<dataset_name>/<log_name>`

请根据本地实验流程自行调整。

## 复现实验建议

当前代码与论文实验的对应关系可以简要理解为：

- **力学反馈在线学习**
  - `online_learning`
- **力学反馈离线评估**
  - `offline_learning`
- **双代价在线学习**
  - `online_learning_dual_cost`
- **数据缓冲区记忆基线**
  - `online_learning_with_databuffer`
- **SALON 风格粗糙度基线**
  - `salon`

推荐流程：

1. 准备包含四类必需输入的数据集根目录；
2. 准备 proprioception checkpoint 与 memory buffer 路径；
3. 先运行 `online_learning` 或 `offline_learning`；
4. 需要双反馈实验时再切换到 `online_learning_dual_cost`。

## 推荐阅读顺序

如果想快速理解当前仓库，建议按以下顺序阅读：

1. `config.py`
2. `polt.py`
3. `polt_runtime/runtime/coordinator.py`
4. `polt_runtime/online_learning.py`
5. `polt_runtime/memory.py`
6. `model/vision/dinov3_infer.py`
7. `model/proprio/TemporalPINNs_Dugoff.py`
8. `model/memory/GPRMemoryForest.py`
9. `polt_runtime/perception.py`
10. `polt_runtime/traversability.py`
11. `sensors/lidar.py`
12. `sensors/proprio.py`
13. `utils/`：离线特征与 VLAD 工具

## 引用方式

如果本仓库对你的研究有帮助，请引用对应论文：

```bibtex
@inproceedings{xie2026polt,
  title     = {Remember Your Driving Feel: Proprioception Based Online Learning for UGV Traversability Across Complex Terrains},
  author    = {Zikang Xie and Binhan Du and Kexin Fei and Shucheng Li and Zhenping Sun and Xiaohui Li and Dewen Hu and Jian Li},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  note      = {Please update publication details according to the final released paper.}
}
```

如需在本地复现实验，请优先根据自己的数据路径、模型权重路径和依赖环境进行调整。
