"""
Post-May 2025 HiRISE patch corpus for SAE training.

Loads pre-sliced patches from cache (PNG or NPY). Does not re-slice or download.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

PATCH_SIZE = 224

_OBS_ID_RE = re.compile(r"^(ESP_\d+_\d+)")


def default_patch_cache_dir() -> Path:
    env = os.environ.get("HIRISE_POST2025_CACHE_DIR", "")
    if env:
        return Path(env)
    if os.name == "nt":
        d_cache = Path(r"D:\hirise_post2025_cache\patches")
        if Path("D:/").exists():
            return d_cache
    from src.paths import RESULTS_DIR

    return RESULTS_DIR / "cache" / "hirise_post2025" / "patches"


def parse_obs_id(directory_name: str) -> str:
    """ESP_088000_1540 from ESP_088000_1540_RED (or full stem)."""
    stem = directory_name.removesuffix(".png").removesuffix(".npy")
    m = _OBS_ID_RE.match(stem)
    if m:
        return m.group(1)
    return stem.split("_RED")[0] if "_RED" in stem else stem


def _scale_to_uint8(patch: np.ndarray) -> np.ndarray:
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if finite.size == 0:
        return np.zeros(patch.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((patch - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _load_patch_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".png":
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        return arr
    return np.load(path)


def scan_patch_cache(cache_dir: Path | None = None) -> List[Dict[str, Any]]:
    """
    Index all cached patches under <cache_dir>/<obs_stem>/.

    Returns flat entries with obs_id, patch_path, image_stem, patch_idx, y, x.
    """
    cache_dir = cache_dir or default_patch_cache_dir()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Post-2025 patch cache not found: {cache_dir}")

    entries: List[Dict[str, Any]] = []
    for obs_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        obs_id = parse_obs_id(obs_dir.name)
        patch_files = sorted(obs_dir.glob("*.png")) + sorted(obs_dir.glob("*.npy"))
        for patch_idx, patch_path in enumerate(patch_files):
            y, x = 0, 0
            parts = patch_path.stem.split("_")
            for i, part in enumerate(parts):
                if part.startswith("y") and part[1:].isdigit():
                    y = int(part[1:])
                if part.startswith("x") and part[1:].isdigit():
                    x = int(part[1:])
            entries.append(
                {
                    "obs_id": obs_id,
                    "image_stem": obs_dir.name,
                    "patch_path": str(patch_path),
                    "cache_path": str(patch_path),
                    "patch_idx": patch_idx,
                    "y": y,
                    "x": x,
                }
            )
    if not entries:
        raise RuntimeError(f"No patch files found under {cache_dir}")
    return entries


class PostMay2025Dataset(Dataset):
    """Cached 224×224 grayscale patches. Returns (patch_tensor, obs_id)."""

    def __init__(
        self,
        entries: Sequence[Dict[str, Any]],
        transform: T.Compose | None = None,
    ):
        self.entries = list(entries)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def obs_ids(self) -> List[str]:
        return [str(e["obs_id"]) for e in self.entries]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        entry = self.entries[idx]
        patch_path = Path(entry["patch_path"])
        patch = _load_patch_array(patch_path)
        if patch.ndim == 3:
            patch = patch[..., 0]
        gray = _scale_to_uint8(np.asarray(patch, dtype=np.float32))
        rgb = np.stack([gray, gray, gray], axis=-1)
        pil = Image.fromarray(rgb, mode="RGB")
        if self.transform is not None:
            tensor = self.transform(pil)
        else:
            tensor = T.ToTensor()(pil)
        return tensor, str(entry["obs_id"])


class PostMay2025IndexedDataset(Dataset):
    """Same patches; returns (tensor, global_idx) for activation matrix indexing."""

    def __init__(
        self,
        entries: Sequence[Dict[str, Any]],
        transform: T.Compose | None = None,
    ):
        self.base = PostMay2025Dataset(entries, transform=transform)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def obs_ids(self) -> List[str]:
        return self.base.obs_ids

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        tensor, _ = self.base[idx]
        return tensor, idx


def split_patch_entries(
    entries: Sequence[Dict[str, Any]],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        by_image.setdefault(e["image_stem"], []).append(e)

    stems = sorted(by_image.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(stems))
    n_train = max(1, int(round(len(stems) * train_fraction)))
    train_stems = {stems[i] for i in perm[:n_train]}
    eval_stems = {stems[i] for i in perm[n_train:]}
    if not eval_stems and len(stems) > 1:
        eval_stems = {stems[perm[-1]]}
        train_stems.discard(next(iter(eval_stems)))

    train_entries = [e for s in train_stems for e in by_image[s]]
    eval_entries = [e for s in eval_stems for e in by_image[s]]
    if not eval_entries:
        eval_entries = train_entries[-max(1, len(train_entries) // 5) :]

    info = {
        "dataset": "post2025_hirise",
        "sae_train_dataset": "post2025_hirise",
        "has_labels": False,
        "train_images": len(train_stems),
        "eval_images": len(eval_stems),
        "train_patches": len(train_entries),
        "eval_patches": len(eval_entries),
        "num_train": len(train_entries),
        "num_eval": len(eval_entries),
        "n_patches": len(entries),
        "n_images_ok": len(stems),
        "cache_dir": str(default_patch_cache_dir()),
    }
    return train_entries, eval_entries, info


def build_post_may2025_datasets(
    cache_dir: Path | None = None,
    transform: T.Compose | None = None,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[
    PostMay2025Dataset,
    PostMay2025Dataset,
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    cache_dir = cache_dir or default_patch_cache_dir()
    entries = scan_patch_cache(cache_dir)
    train_entries, eval_entries, info = split_patch_entries(
        entries, train_fraction=train_fraction, seed=seed
    )
    info["cache_dir"] = str(cache_dir)
    train_ds = PostMay2025Dataset(train_entries, transform=transform)
    eval_ds = PostMay2025Dataset(eval_entries, transform=transform)
    return train_ds, eval_ds, info, train_entries, eval_entries


def validate_post_may2025_cache(cache_dir: Path | None = None) -> Dict[str, Any]:
    cache_dir = cache_dir or default_patch_cache_dir()
    if not cache_dir.is_dir():
        return {
            "dataset": "post2025_hirise",
            "cache_dir": str(cache_dir),
            "n_observations": 0,
            "n_patches": 0,
            "note": f"Cache missing: {cache_dir}",
        }
    obs_dirs = [p for p in cache_dir.iterdir() if p.is_dir()]
    n_patches = sum(
        len(list(d.glob("*.png"))) + len(list(d.glob("*.npy"))) for d in obs_dirs
    )
    return {
        "dataset": "post2025_hirise",
        "sae_train_dataset": "post2025_hirise",
        "cache_dir": str(cache_dir),
        "n_observations": len(obs_dirs),
        "n_patches": n_patches,
        "note": "Load patches from cache only (no re-slice).",
    }
