"""HiRISE v3.2 dataset with configurable train/eval split (real images only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

OFFICIAL_SPLIT_FILENAME = "labels-map-proj-v3_2_train_val_test.txt"
MIN_PER_CLASS_PROBE_REPORT = 20


class HiRISEDataError(RuntimeError):
    """Raised when HiRISE paths or image files are missing or unreadable."""


def is_readable_rgb_image(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Return (True, None) if the file decodes as RGB.
    Does not insert placeholders — only checks readability.
    """
    if not path.is_file():
        return False, "file not found"
    try:
        with Image.open(path) as im:
            im.load()
            im.convert("RGB")
        return True, None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return False, str(exc)


def open_rgb(path: str) -> Image.Image:
    """Load a real RGB image; never substitute placeholders."""
    p = Path(path)
    ok, reason = is_readable_rgb_image(p)
    if not ok:
        raise HiRISEDataError(f"Unreadable image {p}: {reason}")
    with Image.open(p) as im:
        return im.convert("RGB")


def filter_readable_samples(
    samples: List[Dict[str, Any]],
    strict: bool = False,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Drop samples whose files cannot be decoded. In strict mode, raise on first failure."""
    valid: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    iterator = samples
    if show_progress and len(samples) > 0:
        iterator = tqdm(samples, desc="Scan HiRISE images", unit="img")

    for sample in iterator:
        path = Path(sample["path"])
        ok, reason = is_readable_rgb_image(path)
        if ok:
            valid.append(sample)
        elif strict:
            raise HiRISEDataError(
                f"Unreadable HiRISE image (strict mode): {path}\n  reason: {reason}"
            )
        else:
            skipped.append({"rel_path": sample["rel_path"], "reason": reason or "unknown"})

    return valid, skipped


def load_readable_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_readable_cache(cache_path: Path, payload: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_readable_index(
    images_dir: str,
    labels_file: str,
    cache_path: Optional[Path] = None,
    strict: bool = False,
    rescan: bool = False,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]]:
    """
    Parse labels and keep only decodable images.
    Optionally cache results so later runs skip the full scan.
    """
    labels_path = Path(labels_file).resolve()
    images_path = Path(images_dir).resolve()

    if cache_path and not rescan:
        cached = load_readable_cache(cache_path)
        if cached and cached.get("labels_file") == str(labels_path):
            rel_set = set(cached.get("readable_rel_paths", []))
            skipped = cached.get("skipped", [])
            # Rebuild sample dicts from cached rel paths
            valid = []
            for rel in rel_set:
                valid.append(
                    {
                        "path": images_path / rel,
                        "rel_path": rel,
                        "label": cached.get("labels_by_path", {}).get(rel, ""),
                    }
                )
            if valid:
                valid.sort(key=lambda s: s["rel_path"])
                meta = {
                    "images_dir": str(images_path),
                    "labels_file": str(labels_path),
                    "num_listed": cached.get("num_listed", 0),
                    "num_readable": len(valid),
                    "num_skipped": len(skipped),
                    "from_cache": True,
                }
                return valid, skipped, meta

    # Parse all label entries (files that exist on disk)
    ds_tmp = HiRISEDataset.__new__(HiRISEDataset)
    ds_tmp.images_dir = images_path
    ds_tmp.labels_file = labels_path
    ds_tmp.transform = None
    all_samples = ds_tmp._parse_labels_file()
    num_listed = len(all_samples)

    valid, skipped = filter_readable_samples(
        all_samples, strict=strict, show_progress=show_progress
    )
    valid.sort(key=lambda s: s["rel_path"])

    labels_by_path = {s["rel_path"]: s["label"] for s in valid}

    if cache_path:
        save_readable_cache(
            cache_path,
            {
                "images_dir": str(images_path),
                "labels_file": str(labels_path),
                "num_listed": num_listed,
                "num_readable": len(valid),
                "readable_rel_paths": sorted(labels_by_path.keys()),
                "labels_by_path": labels_by_path,
                "skipped": skipped,
            },
        )

    meta = {
        "images_dir": str(images_path),
        "labels_file": str(labels_path),
        "num_listed": num_listed,
        "num_readable": len(valid),
        "num_skipped": len(skipped),
        "from_cache": False,
    }
    return valid, skipped, meta


def load_readable_samples_from_cache(
    cache_path: Path,
    images_dir: str,
    labels_file: str,
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]]]:
    """Fast path: load pre-validated image list from JSON (no per-file PIL open)."""
    cached = load_readable_cache(cache_path)
    if cached is None:
        return None
    labels_path = str(Path(labels_file).resolve())
    images_resolved = str(Path(images_dir).resolve())
    if cached.get("labels_file") != labels_path:
        return None
    if cached.get("images_dir") not in (None, images_resolved):
        return None
    images_path = Path(images_dir)
    labels_by_path = cached.get("labels_by_path", {})
    rel_paths = cached.get("readable_rel_paths", [])
    if not rel_paths:
        return None
    valid = [
        {
            "path": images_path / rel,
            "rel_path": rel,
            "label": labels_by_path.get(rel, ""),
        }
        for rel in rel_paths
    ]
    skipped = cached.get("skipped", [])
    meta = {
        "images_dir": str(images_path.resolve()),
        "labels_file": labels_path,
        "num_listed": cached.get("num_listed", len(valid)),
        "num_readable": len(valid),
        "num_skipped": len(skipped),
        "from_cache": True,
    }
    return valid, skipped, meta


def validate_hirise_paths(
    images_dir: str,
    labels_file: str,
    min_images: int = 1,
    cache_path: Optional[Path] = None,
    strict: bool = False,
    rescan: bool = False,
) -> Dict[str, Any]:
    """
    Verify HiRISE data exists and pre-scan decodable images before any experiment run.
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_file)

    if cache_path and not rescan and not strict:
        cached_load = load_readable_samples_from_cache(
            cache_path, str(images_path), str(labels_path)
        )
        if cached_load is not None:
            valid, skipped, meta = cached_load
            print(
                f"Using cached HiRISE index ({meta['num_readable']} images, "
                f"skipped {meta['num_skipped']} corrupt) — skipping Google Drive path scan.",
                flush=True,
            )
            n = len(valid)
            if n < min_images:
                raise HiRISEDataError(f"Too few readable HiRISE images in cache (count={n}).")
            return {
                "images_dir": meta["images_dir"],
                "labels_file": meta["labels_file"],
                "num_images_listed": meta["num_listed"],
                "num_images_readable": n,
                "num_skipped": len(skipped),
                "note": "Loaded from hirise_readable_cache.json (no Drive rescan).",
            }

    print("Checking HiRISE paths on disk (can be slow on Google Drive)...", flush=True)
    if not images_path.is_dir():
        raise HiRISEDataError(
            f"HiRISE images directory does not exist: {images_path}\n"
            "Set --hirise_images_dir or HIRISE_IMAGES_DIR to your hirise_v3_2/images folder."
        )
    if not labels_path.is_file():
        raise HiRISEDataError(
            f"HiRISE labels file does not exist: {labels_path}\n"
            "Set --hirise_labels_file or HIRISE_LABELS_FILE to labels-map-proj-v3_2.txt."
        )

    print("Scanning HiRISE images for readability (slow first time only)...", flush=True)
    valid, skipped, meta = build_readable_index(
        images_dir=str(images_path),
        labels_file=str(labels_path),
        cache_path=cache_path,
        strict=strict,
        rescan=rescan,
        show_progress=True,
    )

    n = len(valid)
    if n < min_images:
        raise HiRISEDataError(
            f"Too few readable HiRISE images (count={n}, skipped={len(skipped)}). "
            f"Check images under {images_path} or re-download corrupt files."
        )

    note = (
        "Train and eval splits use only decodable real images (80/20). "
        "No placeholders. Unreadable files are excluded, not synthesized."
    )
    if skipped:
        note += f" Skipped {len(skipped)} corrupt/truncated files."

    return {
        "images_dir": str(images_path.resolve()),
        "labels_file": str(labels_path.resolve()),
        "num_images_listed": meta["num_listed"],
        "num_images_readable": n,
        "num_skipped": len(skipped),
        "skipped_sample": skipped[:5],
        "note": note,
    }


class HiRISEDataset(Dataset):
    """Loads HiRISE v3.2 images listed in the official labels file."""

    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        transform=None,
        samples: Optional[List[Dict[str, Any]]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_file = Path(labels_file)
        self.transform = transform
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._parse_labels_file()
        if not self.samples:
            raise HiRISEDataError(
                f"No images in dataset.\n"
                f"  labels: {self.labels_file}\n"
                f"  images: {self.images_dir}"
            )

    def _load_classmap(self) -> Dict[str, str]:
        classmap_path = self.labels_file.parent / "landmarks_map-proj-v3_2_classmap.csv"
        mapping: Dict[str, str] = {}
        if not classmap_path.exists():
            return mapping
        with classmap_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            f.seek(0)
            if first_line and not first_line.lower().startswith("class"):
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "," in line:
                        cid, name = [p.strip() for p in line.split(",", 1)]
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        cid, name = parts[0], " ".join(parts[1:])
                    if cid:
                        mapping[cid] = name
                return mapping
            reader = csv.DictReader(f)
            for row in reader:
                cid = str(row.get("class_id", row.get("id", ""))).strip()
                name = str(
                    row.get("class_name", row.get("name", row.get("semantic_name", "")))
                ).strip()
                if cid and name:
                    mapping[cid] = name
        return mapping

    def _parse_labels_file(self) -> List[Dict[str, Any]]:
        classmap = self._load_classmap()
        samples: List[Dict[str, Any]] = []
        with self.labels_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
                if len(parts) < 2:
                    continue
                rel_path = parts[0]
                label = parts[1]
                if label in classmap:
                    label = classmap[label]
                img_path = self.images_dir / rel_path
                if img_path.is_file():
                    samples.append(
                        {
                            "path": img_path,
                            "rel_path": rel_path,
                            "label": label,
                        }
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = open_rgb(str(sample["path"]))
        if self.transform is not None:
            img = self.transform(img)
        return img, idx


def make_imagenet_transform(
    image_size: int,
    mean: List[float],
    std: List[float],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def split_indices(
    n: int,
    train_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * train_fraction)
    return indices[:split], indices[split:]


def build_hirise_splits(
    images_dir: str,
    labels_file: str,
    transform,
    train_fraction: float = 0.8,
    seed: int = 42,
    readable_samples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Subset, Subset, HiRISEDataset, Dict[str, Any]]:
    dataset = HiRISEDataset(
        images_dir,
        labels_file,
        transform=transform,
        samples=readable_samples,
    )
    train_idx, eval_idx = split_indices(len(dataset), train_fraction, seed)

    split_info = {
        "images_dir": str(Path(images_dir).resolve()),
        "labels_file": str(Path(labels_file).resolve()),
        "seed": seed,
        "train_fraction": train_fraction,
        "num_train": int(len(train_idx)),
        "num_eval": int(len(eval_idx)),
        "num_readable_total": len(dataset),
        "train_rel_paths": [dataset.samples[i]["rel_path"] for i in train_idx],
        "eval_rel_paths": [dataset.samples[i]["rel_path"] for i in eval_idx],
    }
    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, eval_idx.tolist()),
        dataset,
        split_info,
    )


def build_split_info(
    readable_samples: List[Dict[str, Any]],
    train_fraction: float = 0.8,
    seed: int = 42,
    images_dir: str = "",
    labels_file: str = "",
) -> Dict[str, Any]:
    """Build train/eval manifest from pre-validated samples (no dataset instantiation)."""
    samples = sorted(readable_samples, key=lambda s: s["rel_path"])
    train_idx, eval_idx = split_indices(len(samples), train_fraction, seed)
    return {
        "images_dir": str(Path(images_dir).resolve()) if images_dir else "",
        "labels_file": str(Path(labels_file).resolve()) if labels_file else "",
        "seed": seed,
        "train_fraction": train_fraction,
        "num_train": int(len(train_idx)),
        "num_eval": int(len(eval_idx)),
        "num_readable_total": len(samples),
        "train_rel_paths": [samples[i]["rel_path"] for i in train_idx],
        "eval_rel_paths": [samples[i]["rel_path"] for i in eval_idx],
    }


def save_split_manifest(path: Path, split_info: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)


def official_split_path(labels_file: str | Path) -> Path:
    """Path to labels-map-proj-v3_2_train_val_test.txt next to the main labels file."""
    labels_path = Path(labels_file).resolve()
    return labels_path.parent / OFFICIAL_SPLIT_FILENAME


def load_official_split_table(labels_file: str | Path) -> List[Dict[str, str]]:
    """
    Parse official HiRISE v3.2 train/val/test assignments.

    Columns: filename class_id set  (space-separated)
    """
    path = official_split_path(labels_file)
    if not path.is_file():
        raise HiRISEDataError(
            f"Official split file not found: {path}\n"
            "Download labels-map-proj-v3_2_train_val_test.txt from the HiRISE v3.2 Zenodo bundle."
        )
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "rel_path": parts[0],
                    "label": parts[1],
                    "split": parts[2].lower(),
                }
            )
    return rows


def filter_readable_official_samples(
    split_rows: List[Dict[str, str]],
    readable_rel_paths: set[str],
    splits: List[str],
    landforms_only: bool = True,
) -> List[Dict[str, Any]]:
    """Keep rows in requested official splits that are readable on disk."""
    allowed = {s.lower() for s in splits}
    out: List[Dict[str, Any]] = []
    for row in split_rows:
        if row["split"] not in allowed:
            continue
        if landforms_only and row["label"] in ("0", "other", "background", ""):
            continue
        rel = row["rel_path"]
        if rel not in readable_rel_paths:
            continue
        out.append(
            {
                "rel_path": rel,
                "label": row["label"],
                "split": row["split"],
            }
        )
    out.sort(key=lambda s: s["rel_path"])
    return out


def official_split_class_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in samples:
        lab = str(s["label"])
        counts[lab] = counts.get(lab, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99))


def save_official_probe_manifest(
    results_root: Path,
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
) -> Path:
    """Write manifest for official landform probe train/test lists."""
    path = results_root / "official_probe_split_manifest.json"
    payload = {
        "split_file": OFFICIAL_SPLIT_FILENAME,
        "landforms_only": True,
        "expected_test_landforms": 311,
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "train_class_counts": official_split_class_counts(train_samples),
        "test_class_counts": official_split_class_counts(test_samples),
        "train_rel_paths": [s["rel_path"] for s in train_samples],
        "test_rel_paths": [s["rel_path"] for s in test_samples],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
