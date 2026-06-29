#!/usr/bin/env bash
set -euo pipefail

# Create the POLT reproduction environment.
# Run this script from the POLT repository root:
#   bash env/setup_polt_env.sh

ENV_NAME="${ENV_NAME:-polt}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

if [ ! -f "env/requirements-dinov3-cu118.txt" ]; then
  echo "Please run this script from the POLT repository root." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists. Reusing it."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r env/requirements-dinov3-cu118.txt

if [ -d "model/third_party/dinov3" ]; then
  conda run -n "${ENV_NAME}" python -m pip install -e model/third_party/dinov3
else
  echo "Warning: model/third_party/dinov3 was not found. DINOv3 imports may fail." >&2
fi

conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import torch
import torchvision
import numpy
import cv2
import gpytorch
import open3d
import torch_scatter
import fast_pytorch_kmeans
import transformers

print("POLT environment check passed")
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda wheel", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("gpytorch", gpytorch.__version__)
print("open3d", open3d.__version__)
print("torch_scatter", torch_scatter.__version__)
print("transformers", transformers.__version__)
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"
