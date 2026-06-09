"""
Mars-Bench evaluation corpus — pools labeled splits for SAE metrics.

Skips missing splits with a warning. Resizes all images to 224×224.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)

TARGET_SIZE = 224
EVAL_DATASET_NAME = "marsbench"

TaskType = Literal["classification", "segmentation"]


@dataclass(frozen=True)
class MarsBenchSplitSpec:
    key: str
    task: TaskType
    hf_repo: str
    local_dir_names: Tuple[str, ...]


MARS_BENCH_EVAL_SPLITS: Tuple[MarsBenchSplitSpec, ...] = (
    MarsBenchSplitSpec(
        "AtmosDust",
        "classification",
        "Mirali33/mb-atmospheric_dust_cls_rdr",
        ("atmospheric_dust_classification_rdr", "mb-atmospheric_dust_cls_rdr", "AtmosDust"),
    ),
    MarsBenchSplitSpec(
        "DoMars16k",
        "classification",
        "Mirali33/mb-domars16k",
        ("domars16k", "mb-domars16k", "DoMars16k"),
    ),
    MarsBenchSplitSpec(
        "Frost",
        "classification",
        "Mirali33/mb-frost_cls",
        ("frost_classification", "mb-frost_cls", "Frost"),
    ),
    MarsBenchSplitSpec(
        "Landmark",
        "classification",
        "Mirali33/mb-landmark_cls",
        ("landmark_classification", "mb-landmark_cls", "Landmark"),
    ),
    MarsBenchSplitSpec(
        "Boulder",
        "segmentation",
        "Mirali33/mb-boulder_seg",
        ("boulder_segmentation", "mb-boulder_seg", "Boulder"),
    ),
    MarsBenchSplitSpec(
        "ConeQuest",
        "segmentation",
        "Mirali33/mb-conequest_seg",
        ("conequest_segmentation", "mb-conequest_seg", "ConeQuest"),
    ),
    MarsBenchSplitSpec(
        "CraterBinary",
        "segmentation",
        "Mirali33/mb-crater_binary_seg",
        ("crater_binary_segmentation", "mb-crater_binary_seg", "CraterBinary"),
    ),
    MarsBenchSplitSpec(
        "CraterMulti",
        "segmentation",
        "Mirali33/mb-crater_multi_seg",
        ("crater_multi_segmentation", "mb-crater_multi_seg", "CraterMulti"),
    ),
    MarsBenchSplitSpec(
        "MMLS",
        "segmentation",
        "Mirali33/mb-mmls",
        ("mmls", "mb-mmls", "MMLS"),
    ),
)


def default_marsbench_root() -> Path:
    env = os.environ.get("MARS_BENCH_ROOT", "")
    if env:
        return Path(env)
    if os.name == "nt":
        candidates = [
            Path(r"D:\Mars-Bench"),
            Path(r"D:\mars_bench"),
            Path(r"G:\My Drive\Mars-Bench"),
            Path(r"G:\My Drive\metrics&models\Mars-Bench"),
        ]
        for c in candidates:
            if c.is_dir():
                return c
    from src.paths import DEFAULT_PATHS

    return Path(DEFAULT_PATHS["drive_root"]) / "Mars-Bench"


def _resize_rgb(pil: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BILINEAR)
    return pil


def _label_from_mask(mask: Any, multi: bool = False) -> str:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    positive = arr > 0
    if not positive.any():
        return "0"
    if not multi:
        return "1"
    vals, counts = np.unique(arr[positive], return_counts=True)
    return str(int(vals[counts.argmax()]))


def _load_hf_split(spec: MarsBenchSplitSpec, split: str) -> List[Dict[str, Any]] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning(
            "datasets package not installed; cannot load Mars-Bench split %s from HF",
            spec.key,
        )
        return None
    try:
        ds = load_dataset(spec.hf_repo, split=split)
    except Exception as exc:
        logger.warning("Skipping Mars-Bench %s split=%s (HF): %s", spec.key, split, exc)
        return None

    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        row = ds[i]
        image = row.get("image") or row.get("img") or row.get("pixel_values")
        if image is None:
            continue
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = _resize_rgb(image)

        if spec.task == "classification":
            label = row.get("label", row.get("class", row.get("category", 0)))
            label_str = str(int(label) if isinstance(label, (int, np.integer)) else label)
        else:
            mask = row.get("mask") or row.get("label") or row.get("segmentation")
            if mask is None:
                continue
            label_str = _label_from_mask(mask, multi=spec.key == "CraterMulti")

        rows.append(
            {
                "dataset_name": spec.key,
                "split": split,
                "label": label_str,
                "label_key": f"{spec.key}:{label_str}",
                "image": image,
                "index": i,
            }
        )
    return rows


def _load_local_split(
    root: Path, spec: MarsBenchSplitSpec, split: str
) -> List[Dict[str, Any]] | None:
    base: Path | None = None
    for name in spec.local_dir_names:
        candidate = root / name
        if candidate.is_dir():
            base = candidate
            break
        candidate = root / name.lower()
        if candidate.is_dir():
            base = candidate
            break
    if base is None:
        return None

    img_dir = base / split / "images"
    if not img_dir.is_dir():
        img_dir = base / split
    if not img_dir.is_dir():
        return None

    rows: List[Dict[str, Any]] = []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        image = _resize_rgb(Image.open(img_path).convert("RGB"))
        label_str = "0"
        mask_path = base / split / "masks" / f"{img_path.stem}.png"
        if mask_path.is_file():
            label_str = _label_from_mask(
                np.asarray(Image.open(mask_path)), multi=spec.key == "CraterMulti"
            )
        else:
            cls_path = base / split / "labels" / f"{img_path.stem}.txt"
            if cls_path.is_file():
                label_str = cls_path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "dataset_name": spec.key,
                "split": split,
                "label": label_str,
                "label_key": f"{spec.key}:{label_str}",
                "image": image,
                "path": str(img_path),
            }
        )
    return rows if rows else None


def load_marsbench_pool(
    splits: Sequence[str] = ("train", "val", "test"),
    root: Path | None = None,
    specs: Sequence[MarsBenchSplitSpec] = MARS_BENCH_EVAL_SPLITS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = root or default_marsbench_root()
    pool: List[Dict[str, Any]] = []
    loaded_splits: List[str] = []
    skipped: List[str] = []

    for spec in specs:
        spec_rows: List[Dict[str, Any]] = []
        for split in splits:
            local_rows = _load_local_split(root, spec, split) if root.is_dir() else None
            if local_rows:
                spec_rows.extend(local_rows)
            else:
                hf_rows = _load_hf_split(spec, split)
                if hf_rows:
                    spec_rows.extend(hf_rows)
        if spec_rows:
            pool.extend(spec_rows)
            loaded_splits.append(spec.key)
        else:
            skipped.append(spec.key)
            logger.warning("Skipping Mars-Bench split %s (not on disk / HF unavailable)", spec.key)

    summary = {
        "eval_dataset": EVAL_DATASET_NAME,
        "marsbench_root": str(root),
        "loaded_splits": loaded_splits,
        "skipped_splits": skipped,
        "n_samples": len(pool),
        "target_size": TARGET_SIZE,
    }
    return pool, summary


class MarsBenchDataset(Dataset):
    """Pooled Mars-Bench samples. Returns (image_tensor, label_key, dataset_name)."""

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
    def labels(self) -> List[str]:
        return [str(e["label_key"]) for e in self.entries]

    @property
    def dataset_names(self) -> List[str]:
        return [str(e["dataset_name"]) for e in self.entries]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        entry = self.entries[idx]
        pil = entry["image"]
        if self.transform is not None:
            tensor = self.transform(pil)
        else:
            tensor = T.ToTensor()(pil)
        return tensor, str(entry["label_key"]), str(entry["dataset_name"])


class MarsBenchProbeDataset(Dataset):
    """Returns (tensor, label_key) for existing activation / probe pipelines."""

    def __init__(
        self,
        entries: Sequence[Dict[str, Any]],
        transform: T.Compose | None = None,
    ):
        self.base = MarsBenchDataset(entries, transform=transform)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def labels(self) -> List[str]:
        return self.base.labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        tensor, label_key, _ = self.base[idx]
        return tensor, label_key


def _subsample_pool(
    pool: List[Dict[str, Any]],
    max_samples: int | None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(pool) <= max_samples:
        return pool
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(pool), size=max_samples, replace=False)
    return [pool[int(i)] for i in sorted(idxs)]


def build_marsbench_probe_splits(
    transform: T.Compose | None = None,
    root: Path | None = None,
    train_splits: Sequence[str] = ("train", "val"),
    eval_splits: Sequence[str] = ("test",),
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    seed: int = 42,
) -> Tuple[MarsBenchProbeDataset, MarsBenchProbeDataset, Dict[str, Any]]:
    train_pool, train_summary = load_marsbench_pool(splits=train_splits, root=root)
    eval_pool, eval_summary = load_marsbench_pool(splits=eval_splits, root=root)
    train_pool = _subsample_pool(train_pool, max_train_samples, seed=seed)
    eval_pool = _subsample_pool(eval_pool, max_eval_samples, seed=seed + 1)
    if not train_pool:
        raise RuntimeError("No Mars-Bench train samples loaded (all splits missing?)")
    if not eval_pool:
        logger.warning("No Mars-Bench eval samples; falling back to train pool subset")
        eval_pool = train_pool[-max(1, len(train_pool) // 5) :]

    train_ds = MarsBenchProbeDataset(train_pool, transform=transform)
    eval_ds = MarsBenchProbeDataset(eval_pool, transform=transform)
    from collections import Counter

    eval_counts = Counter(e["label_key"] for e in eval_pool)
    summary = {
        "eval_dataset": EVAL_DATASET_NAME,
        "train_samples": len(train_pool),
        "eval_samples": len(eval_pool),
        "loaded_train_splits": train_summary["loaded_splits"],
        "loaded_eval_splits": eval_summary["loaded_splits"],
        "skipped_splits": sorted(
            set(train_summary["skipped_splits"]) | set(eval_summary["skipped_splits"])
        ),
        "eval_class_counts": {k: int(v) for k, v in eval_counts.items()},
    }
    return train_ds, eval_ds, summary
