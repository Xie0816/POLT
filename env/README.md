# POLT Environment Setup

This folder contains the recommended environment files for reproducing POLT with
the `polt` conda environment.

## Tested Environment

The current POLT code was tested with:

| Component | Version |
| --- | --- |
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

The machine should have an NVIDIA driver compatible with CUDA 11.8 PyTorch
wheels. Check with:

```bash
nvidia-smi
```

## Quick Install

Run from the POLT repository root:

```bash
bash env/setup_polt_env.sh
conda activate polt
python polt.py --help
```

The script creates the `polt` conda environment, installs the pinned pip
dependencies, installs the local DINOv3 source under `model/third_party/dinov3`,
and runs a small import/version check.

## Manual Install

If you prefer manual setup:

```bash
conda create -y -n polt python=3.11 pip
conda activate polt
python -m pip install --upgrade pip
python -m pip install -r env/requirements-dinov3-cu118.txt
python -m pip install -e model/third_party/dinov3
python polt.py --help
```

## Files

- `environment.yml`: minimal conda environment descriptor.
- `requirements-dinov3-cu118.txt`: pinned Python packages for the tested CUDA
  11.8 PyTorch stack.
- `setup_polt_env.sh`: reproducible setup script.

## Notes

- POLT loads DINOv3 source from `model/third_party/dinov3`; keep this directory
  in the repository or reinstall DINOv3 from the official source:
  <https://github.com/facebookresearch/dinov3>.
- The default DINOv3 checkpoint is expected at
  `weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`.
  Download it from the official DINOv3 access page and save it with this exact
  filename:
  <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>.
- Model weights and dataset paths are not installed by the environment script.
  Please adjust paths according to your local setup.
- If Matplotlib reports that `~/.config/matplotlib` is not writable, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```
