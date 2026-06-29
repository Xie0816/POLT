# POLT: Proprioception-based Online Learning for UGV Traversability

## Project Overview

This repository contains the core codebase associated with **POLT**.

POLT targets **UGV traversability prediction in complex off-road environments**, where visual appearance and true physical traversability may be inconsistent. The framework uses **proprioceptive feedback for online learning**, and combines it with a **dynamic memory mechanism** to accumulate experience and predict terrain traversability costs ahead of the vehicle.

The current repository is organized around the main experimental pipeline of the paper:

- sensor processing,
- proprioception-driven feedback estimation,
- online/offline learning,
- feature-point-cloud fusion,
- dynamic memory based traversability prediction.

Note: `planning/` is included as an optional extension for the dual-cost preset. The main focus of this release is the **perception + online learning** pipeline.

## Features

- Unified runtime entry through `polt.py`
- Unified experiment configuration through `config.py`
- Support for **5 paper-related experiment presets**
- LiDAR, image, and proprioceptive processing modules
- Online learning with:
  - `dynamic_memory`
  - `max_similiarity_out`
- Single-feedback and dual-feedback traversability pipelines
- Logging and runtime profiling utilities

## Repository Structure

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

### Code map

- `polt.py`: command-line entrypoint
- `config.py`: preset definitions and runtime configuration
- `sensors/`: sensor-level preprocessing
- `polt_runtime/online_learning.py`: main experiment pipeline
- `polt_runtime/memory.py`: memory backends
- `polt_runtime/perception.py`: multi-sensor synchronization and feature-frame construction
- `polt_runtime/traversability.py`: feedback estimation and traversability statistics
- `polt_runtime/runtime/`: runtime orchestration, dataset loading, frame-level helpers, profiling, analysis
- `model/vision/`: DINOv3 inference wrapper and VLAD feature utilities
- `model/proprio/`: proprioceptive mechanical model and SALON roughness cost model
- `model/memory/`: dynamic memory and data-buffer GPR implementations
- `model/third_party/`: vendored third-party model source, currently DINOv3
- `planning/`: optional dual-cost risk-map construction, MPPI backend, and planner visualization
- `utils/`: offline DINO feature export and VLAD training utilities

Legacy DINO project outputs that are not part of the runtime model API are kept under `outputs/legacy_dino_project/`.

## Installation

The recommended environment files are provided under `env/`. The default conda
environment name is `polt`.

The code is expected to run on **Linux + NVIDIA GPU + CUDA-compatible driver**.
The tested environment uses the CUDA 11.8 PyTorch wheel stack.

### Option 1: one-command setup

Run from the repository root:

```bash
bash env/setup_polt_env.sh
conda activate polt
python polt.py --help
```

The script will:

- create the `polt` conda environment;
- install the pinned Python dependencies from `env/requirements-dinov3-cu118.txt`;
- install the local DINOv3 source from `model/third_party/dinov3`;
- run a small import/version check.

### Option 2: manual setup

```bash
conda create -y -n polt python=3.11 pip
conda activate polt
python -m pip install --upgrade pip
python -m pip install -r env/requirements-dinov3-cu118.txt
python -m pip install -e model/third_party/dinov3
python polt.py --help
```

### Tested core versions

| Component | Version |
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

For more details, see `env/README.md`.

Before installation, make sure your NVIDIA driver is available:

```bash
nvidia-smi
```

Model weights, datasets, and vendored DINOv3 source are not committed to this
GitHub repository. Prepare the DINOv3 source and weights as described below, and
place the dataset according to your local setup.

## DINOv3 Source and Weights

POLT uses DINOv3 as the visual feature backbone. The current runtime loads it
through `model/vision/dinov3_infer.py` with:

- source code: `model/third_party/dinov3`
- default model key: `dinov3_vits16plus`
- default checkpoint:
  `weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`

The GitHub repository does not include a vendored copy of DINOv3. Install the
official implementation manually before running POLT:

```bash
git clone https://github.com/facebookresearch/dinov3.git model/third_party/dinov3
python -m pip install -e model/third_party/dinov3
```

DINOv3 checkpoints are large and may be subject to the original DINOv3 license
and access process, so they are not committed to this repository or downloaded
automatically. To prepare the default POLT checkpoint:

1. Open the official DINOv3 download page:
   <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>
2. Request access and accept the required license terms.
3. Use the URL provided by the DINOv3 access email to download the
   **ViT-S+/16 distilled, LVD-1689M** checkpoint.
4. Save it with the exact filename expected by POLT:

```bash
mkdir -p weights/dinov3
wget "<URL_FROM_DINOV3_ACCESS_EMAIL>" \
  -O weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

The repository also supports keeping the optional ViT-S/16 checkpoint under:

```text
weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

The official DINOv3 models are also listed in the Hugging Face collection:
<https://huggingface.co/collections/facebook/dinov3>. The current POLT runtime,
however, expects a local `.pth` checkpoint passed to `torch.hub.load(...,
source="local", weights=...)`; using Hugging Face loading directly requires
adapting `model/vision/dinov3_infer.py`.

Expected layout:

```text
POLT/
├── model/
│   └── third_party/
│       └── dinov3/
└── weights/
    └── dinov3/
        ├── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
        └── dinov3_vits16_pretrain_lvd1689m-08c60483.pth  # optional
```

## Dataset Preparation

`--root` must point to the **dataset root directory**.

From the runtime code in `polt_runtime/runtime/data.py`, the following inputs are required under `root`:

- `image_front/`
- `lidar/`
- `lidarodometry.txt`
- `proprio_infos.json`

Recommended layout:

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

Notes:

- Image and LiDAR filenames are expected to use **timestamp-based numeric stems**.
- `lidarodometry.txt` is loaded as numeric text.
- `proprio_infos.json` is loaded as a timestamp-keyed JSON dictionary.
- Please adjust according to your local setup.

Important:

- `polt.py` currently contains a developer-local default value for `--root`.
- In practice, you should **always pass `--root` explicitly**.

## Quick Start

### Show CLI help

```bash
conda run -n polt python polt.py --help
```

### Run `online_learning`

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset online_learning
```

### Run `offline_learning`

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset offline_learning
```

### Run `online_learning_dual_cost`

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset online_learning_dual_cost \
  --mechanical-mem-buffer /path/to/mechanical_buffer \
  --roughness-mem-buffer /path/to/roughness_buffer
```

If you want to disable the planner-related branch in dual-cost mode:

```bash
conda run -n polt python polt.py \
  --root /path/to/dataset_root \
  --preset online_learning_dual_cost \
  --mechanical-mem-buffer /path/to/mechanical_buffer \
  --roughness-mem-buffer /path/to/roughness_buffer \
  --disable-planner
```

## Experiment Presets

The repository currently exposes the following preset names from `config.py`.

| Preset | Feedback | Memory | Update | Default log file | Notes |
|---|---|---|---|---|---|
| `offline_learning` | `mechanical` | `dynamic_memory` | `offline` | `log_offline.json` | Offline evaluation without online update |
| `online_learning` | `mechanical` | `dynamic_memory` | `online` | `log.json` | Main single-feedback online setting |
| `online_learning_dual_cost` | `both` | `dynamic_memory` | `online` | `log_dual_cost.json` | Dual-feedback setting; planner enabled by default |
| `online_learning_with_databuffer` | `mechanical` | `max_similiarity_out` | `online` | `log_databuffer.json` | Buffer-style memory baseline |
| `salon` | `roughness` | `max_similiarity_out` | `online` | `log_salon.json` | Roughness-oriented baseline |

### Main configuration axes

- `feedback_mode`
  - `mechanical`
  - `roughness`
  - `both`
- `memory_mode`
  - `dynamic_memory`
  - `max_similiarity_out`
- `update_mode`
  - `online`
  - `offline`

Note:

- The code still accepts the legacy internal alias `fixed_size_data_buffer`.
- The public name used by the current codebase is `max_similiarity_out`.

## Outputs

By default, logs are written according to the active preset:

- `log.json`
- `log_offline.json`
- `log_databuffer.json`
- `log_salon.json`
- `log_dual_cost.json`

Output location depends on the mode:

- For presets without planner output, logs are typically saved under:
  - `<root>/<log_name>`
- For `online_learning_dual_cost`, if planner remains enabled, the runtime may save logs under:
  - `planner_traces/<dataset_name>/<log_name>`

Please adjust your workflow according to your local setup.

## Reproducing Paper Experiments

For the current codebase, the most direct mapping from paper experiments to runtime presets is:

- **Mechanical online learning**:
  - `online_learning`
- **Mechanical offline evaluation**:
  - `offline_learning`
- **Dual-cost online learning**:
  - `online_learning_dual_cost`
- **Data-buffer memory baseline**:
  - `online_learning_with_databuffer`
- **SALON-style roughness baseline**:
  - `salon`

Suggested workflow:

1. Prepare the dataset root with the required four inputs.
2. Prepare the proprioception checkpoint and memory-buffer paths.
3. Start from `online_learning` or `offline_learning`.
4. Move to `online_learning_dual_cost` when dual-feedback experiments are needed.

## Recommended Reading Order

If you want to understand the repository quickly, the recommended reading order is:

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
13. `utils/` for offline feature and VLAD tools

## Citation

If you find this repository useful, please cite the corresponding paper.

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
