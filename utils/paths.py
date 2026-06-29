"""Path helpers shared by POLT utility scripts."""

from __future__ import annotations

import sys
from pathlib import Path


POLT_ROOT = Path(__file__).resolve().parents[1]
DINO_ROOT = POLT_ROOT / "model" / "vision"
DINO_SOURCE_ROOT = POLT_ROOT / "model" / "third_party" / "dinov3"
DINO_CUDA_OPS_ROOT = DINO_SOURCE_ROOT / "dinov3" / "eval" / "segmentation" / "models" / "utils" / "ops"


def add_path(path: Path) -> None:
    path_text = str(path)
    if path.exists() and path_text not in sys.path:
        sys.path.insert(0, path_text)


def setup_dino_project_paths() -> None:
    """Expose local DINOv3 source and project data helpers to standalone scripts."""
    for path in (DINO_CUDA_OPS_ROOT, DINO_SOURCE_ROOT, DINO_ROOT, POLT_ROOT):
        add_path(path)
