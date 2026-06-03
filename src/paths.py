"""Path resolution for HiRISE data and local checkpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


def get_default_paths() -> Dict[str, str]:
    if sys.platform == "win32":
        drive_root = os.environ.get(
            "MARS_DRIVE_ROOT", r"G:\My Drive\metrics&models"
        )
    else:
        drive_root = os.environ.get(
            "MARS_DRIVE_ROOT",
            "/Users/kacy/Library/CloudStorage/"
            "GoogleDrive-kacy.morgan.hat@gmail.com/My Drive/metrics&models",
        )

    project_root = Path(drive_root) / "mars_dino_project"
    mars_data_root = Path(drive_root) / "mars_data"
    hf_home = Path(drive_root) / "hf_cache"

    return {
        "drive_root": str(drive_root),
        "mars_orbital_vit_checkpoint": str(
            project_root / "checkpoints" / "best_backbone.pt"
        ),
        "momo_checkpoint": os.environ.get("MOMO_CHECKPOINT_PATH", ""),
        "hirise_root": str(mars_data_root / "hirise_v3_2"),
        "hirise_images_dir": str(mars_data_root / "hirise_v3_2" / "images"),
        "hirise_labels_file": str(
            mars_data_root / "hirise_v3_2" / "labels-map-proj-v3_2.txt"
        ),
        "hf_home": str(hf_home),
    }


DEFAULT_PATHS = get_default_paths()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOCAL_HF_CACHE = PROJECT_ROOT / ".hf_cache"


def resolve_hf_home() -> str:
    """Use Drive hf_cache when available; else project-local .hf_cache."""
    configured = os.environ.get("HF_HOME") or DEFAULT_PATHS.get("hf_home", "")
    if configured:
        drive_parent = Path(configured).parent
        if drive_parent.exists():
            return configured
    return str(LOCAL_HF_CACHE)
