"""
HiRISE post-May 2025 JP2 dataset (unsupervised patch inspection).

Reads RED JP2 tiles, slices non-overlapping 224×224 patches, caches per image.
No class labels — not for probe / F1 evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T
from tqdm import tqdm

PATCH_SIZE = 224
MAX_NODATA_FRAC = 0.10


def default_post2025_dir() -> Path:
    env = os.environ.get("HIRISE_POST2025_DIR", "")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(r"G:\My Drive\hirise_post_may2025")
    return Path(
        "/Users/kacy/Library/CloudStorage/"
        "GoogleDrive-kacy.morgan.hat@gmail.com/My Drive/hirise_post_may2025"
    )


def default_patch_cache_dir() -> Path:
    env = os.environ.get("HIRISE_POST2025_CACHE_DIR", "")
    if env:
        return Path(env)
    # Prefer D: when available — full-scene .npy stacks can be 100s of MB per JP2.
    if os.name == "nt":
        d_cache = Path(r"D:\hirise_post2025_cache\patches")
        if Path("D:/").exists():
            return d_cache
    from src.paths import RESULTS_DIR

    return RESULTS_DIR / "cache" / "hirise_post2025" / "patches"


def default_slices_dir() -> Path:
    """Directory for per-patch PNG exports (viewable slices)."""
    env = os.environ.get("HIRISE_POST2025_SLICES_DIR", "")
    if env:
        return Path(env)
    if os.name == "nt" and Path("D:/").exists():
        return Path(r"D:\hirise_post2025_cache\slices")
    from src.paths import RESULTS_DIR

    return RESULTS_DIR / "cache" / "hirise_post2025" / "slices"


def image_patch_dir(cache_dir: Path, image_stem: str) -> Path:
    return cache_dir / image_stem


def patch_npy_path(
    cache_dir: Path, image_stem: str, patch_idx: int, y: int, x: int
) -> Path:
    return image_patch_dir(cache_dir, image_stem) / f"patch_{patch_idx:04d}_y{y}_x{x}.npy"


def default_manifest_path() -> Path:
    from src.paths import RESULTS_DIR

    return RESULTS_DIR / "cache" / "hirise_post2025" / "patch_manifest.csv"


def parse_esp_metadata(jp2_path: Path) -> Dict[str, str]:
    """
    Extract HiRISE ESP IDs from filenames like ESP_088000_2615_RED.JP2.

    Orbit number maps to acquisition era (post-May-2025 orbits in download script).
    Full UTC timestamps require PDS label files — not bundled in RED JP2 alone.
    """
    stem = jp2_path.stem
    parts = stem.split("_")
    meta: Dict[str, str] = {"esp_id": stem.replace("_RED", "")}
    if len(parts) >= 3 and parts[0] == "ESP":
        meta["orbit_number"] = parts[1]
        meta["observation_id"] = parts[2]
        meta["orb_folder"] = f"ORB_{parts[1][:3]}00_{parts[1][:3]}99"
    return meta


def list_jp2_files(source_dir: Path) -> List[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"HiRISE post-2025 source dir not found: {source_dir}")
    files = sorted(source_dir.glob("*.JP2")) + sorted(source_dir.glob("*.jp2"))
    if not files:
        raise FileNotFoundError(f"No JP2 files under {source_dir}")
    return files


def read_jp2_band(path: Path) -> np.ndarray:
    """Read first band from a JP2 file as float32 H×W."""
    errors: list[str] = []
    try:
        import glymur

        data = glymur.Jp2k(str(path))[:]
        arr = np.asarray(data, dtype=np.float32)
    except ImportError:
        errors.append("glymur not installed")
        arr = None
    except Exception as exc:
        errors.append(f"glymur failed: {exc}")
        arr = None

    if arr is None:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "JP2 reading requires glymur or rasterio. Install with:\n"
                '  pip install glymur rasterio'
            ) from exc
        try:
            with rasterio.open(path) as src:
                arr = np.asarray(src.read(1), dtype=np.float32)
        except Exception as exc:
            detail = "; ".join(errors + [f"rasterio failed: {exc}"])
            raise RuntimeError(f"Could not read JP2 {path}: {detail}") from exc
    if arr.ndim != 2:
        arr = arr[0] if arr.ndim == 3 else arr.squeeze()
    return arr


def nodata_fraction(patch: np.ndarray) -> float:
    """Fraction of nodata / zero pixels in a patch."""
    if patch.size == 0:
        return 1.0
    invalid = (patch <= 0) | ~np.isfinite(patch)
    return float(invalid.mean())


def slice_patches(
    image: np.ndarray,
    patch_size: int = PATCH_SIZE,
    max_nodata_frac: float = MAX_NODATA_FRAC,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """
    Return (n_patches, patch_size, patch_size) float32 array and grid metadata.

    Metadata entries: {patch_idx, y, x} pixel offsets in the source image.
    """
    h, w = image.shape[:2]
    patches: List[np.ndarray] = []
    meta: List[Dict[str, int]] = []
    patch_idx = 0
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image[y : y + patch_size, x : x + patch_size]
            if nodata_fraction(patch) > max_nodata_frac:
                continue
            patches.append(patch.astype(np.float32, copy=False))
            meta.append({"patch_idx": patch_idx, "y": y, "x": x})
            patch_idx += 1
    if not patches:
        return np.empty((0, patch_size, patch_size), dtype=np.float32), []
    return np.stack(patches, axis=0), meta


def cache_path_for_image(cache_dir: Path, jp2_path: Path) -> Path:
    """Legacy monolithic stack path (deprecated; per-patch dirs used instead)."""
    return cache_dir / f"{jp2_path.stem}.npy"


def _load_cached_patches(
    jp2_path: Path, cache_dir: Path, meta_path: Path
) -> Tuple[np.ndarray, List[Dict[str, int]]] | None:
    if not meta_path.is_file():
        return None
    with meta_path.open(encoding="utf-8") as f:
        cached_meta = json.load(f)
    grid_meta = cached_meta.get("grid", [])
    if not grid_meta:
        return np.empty((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32), []

    stem = jp2_path.stem
    if cached_meta.get("storage") == "per_patch":
        patches = []
        for gm in grid_meta:
            ppath = patch_npy_path(cache_dir, stem, gm["patch_idx"], gm["y"], gm["x"])
            if not ppath.is_file():
                return None
            patches.append(np.load(ppath))
        return np.stack(patches, axis=0), grid_meta

    legacy = cache_path_for_image(cache_dir, jp2_path)
    if legacy.is_file():
        return np.load(legacy), grid_meta
    return None


def _save_patches_per_file(
    patches: np.ndarray,
    grid_meta: List[Dict[str, int]],
    cache_dir: Path,
    image_stem: str,
) -> List[str]:
    out_dir = image_patch_dir(cache_dir, image_stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for i, gm in enumerate(grid_meta):
        ppath = patch_npy_path(cache_dir, image_stem, gm["patch_idx"], gm["y"], gm["x"])
        np.save(ppath, patches[i])
        paths.append(str(ppath))
    return paths


def slice_png_path(slices_dir: Path, image_stem: str, patch_idx: int, y: int, x: int) -> Path:
    """PNG path: slices/<stem>/patch_{idx:04d}_y{y}_x{x}.png"""
    return slices_dir / image_stem / f"patch_{patch_idx:04d}_y{y}_x{x}.png"


def save_patch_pngs(
    patches: np.ndarray,
    grid_meta: List[Dict[str, int]],
    image_stem: str,
    slices_dir: Path,
) -> List[str]:
    """Write viewable RGB PNGs for each patch; return relative paths."""
    if patches.size == 0:
        return []
    out_dir = slices_dir / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: List[str] = []
    for i in range(patches.shape[0]):
        gm = grid_meta[i]
        gray = _scale_to_uint8(patches[i])
        rgb = np.stack([gray, gray, gray], axis=-1)
        png_path = slice_png_path(slices_dir, image_stem, gm["patch_idx"], gm["y"], gm["x"])
        Image.fromarray(rgb, mode="RGB").save(png_path, format="PNG")
        rel_paths.append(str(png_path.relative_to(slices_dir.parent)))
    return rel_paths


def extract_and_cache_patches(
    jp2_path: Path,
    cache_dir: Path,
    patch_size: int = PATCH_SIZE,
    max_nodata_frac: float = MAX_NODATA_FRAC,
    force: bool = False,
    slices_dir: Path | None = None,
    save_png_slices: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{jp2_path.stem}.json"
    stem = jp2_path.stem

    if not force:
        loaded = _load_cached_patches(jp2_path, cache_dir, meta_path)
        if loaded is not None:
            patches, grid_meta = loaded
            if save_png_slices and slices_dir is not None and grid_meta:
                png_dir = slices_dir / stem
                if not png_dir.is_dir():
                    save_patch_pngs(patches, grid_meta, stem, slices_dir)
            return patches, grid_meta

    image = read_jp2_band(jp2_path)
    patches, grid_meta = slice_patches(image, patch_size, max_nodata_frac)
    patch_paths = _save_patches_per_file(patches, grid_meta, cache_dir, stem)
    png_paths: List[str] = []
    if save_png_slices and slices_dir is not None and patches.size:
        png_paths = save_patch_pngs(patches, grid_meta, stem, slices_dir)
    meta = {
        "source": str(jp2_path),
        "n_patches": int(patches.shape[0]),
        "patch_size": patch_size,
        "max_nodata_frac": max_nodata_frac,
        "storage": "per_patch",
        "patch_dir": str(image_patch_dir(cache_dir, stem)),
        "patch_files": patch_paths,
        "grid": grid_meta,
        "png_slices": png_paths,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return patches, grid_meta


def build_patch_index(
    source_dir: Path,
    cache_dir: Path,
    patch_size: int = PATCH_SIZE,
    max_nodata_frac: float = MAX_NODATA_FRAC,
    force_reslice: bool = False,
    show_progress: bool = True,
    slices_dir: Path | None = None,
    save_png_slices: bool = True,
    manifest_path: Path | None = None,
    strict: bool = False,
    skipped_log: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build flat patch index across all JP2 images.

    Each entry: {image_stem, patch_idx, cache_path, global_idx, y, x, png_path?}
    Corrupt / undecodable JP2s are skipped by default (logged to skipped_log).
    """
    import csv

    jp2_files = list_jp2_files(source_dir)
    if slices_dir is None and save_png_slices:
        slices_dir = default_slices_dir()
    entries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    total_patches = 0
    iterator = jp2_files
    if show_progress:
        iterator = tqdm(jp2_files, desc="Slice/cache JP2 patches", unit="image")

    for jp2_path in iterator:
        try:
            patches, grid_meta = extract_and_cache_patches(
                jp2_path,
                cache_dir,
                patch_size=patch_size,
                max_nodata_frac=max_nodata_frac,
                force=force_reslice,
                slices_dir=slices_dir,
                save_png_slices=save_png_slices,
            )
        except Exception as exc:
            if strict:
                raise
            skipped.append({"path": str(jp2_path), "error": str(exc)})
            if show_progress:
                tqdm.write(f"[SKIP] {jp2_path.name}: {exc}")
            continue
        for local_i, gm in enumerate(grid_meta):
            png_path = None
            if save_png_slices and slices_dir is not None:
                png_path = str(
                    slice_png_path(
                        slices_dir,
                        jp2_path.stem,
                        gm["patch_idx"],
                        gm["y"],
                        gm["x"],
                    )
                )
            patch_file = str(
                patch_npy_path(
                    cache_dir, jp2_path.stem, gm["patch_idx"], gm["y"], gm["x"]
                )
            )
            entries.append(
                {
                    "image_stem": jp2_path.stem,
                    "source_jp2": str(jp2_path),
                    **parse_esp_metadata(jp2_path),
                    "patch_idx": gm["patch_idx"],
                    "y": gm["y"],
                    "x": gm["x"],
                    "patch_file": patch_file,
                    "cache_path": patch_file,
                    "png_path": png_path,
                    "global_idx": total_patches + local_i,
                }
            )
        total_patches += int(patches.shape[0])

    summary = {
        "source_dir": str(source_dir),
        "cache_dir": str(cache_dir),
        "slices_dir": str(slices_dir) if slices_dir else "",
        "n_images": len(jp2_files),
        "n_images_ok": len(jp2_files) - len(skipped),
        "n_images_skipped": len(skipped),
        "n_patches": total_patches,
        "patch_size": patch_size,
        "max_nodata_frac": max_nodata_frac,
        "save_png_slices": save_png_slices,
    }
    if skipped:
        summary["skipped_sample"] = skipped[:3]
        if skipped_log is not None:
            skipped_log.parent.mkdir(parents=True, exist_ok=True)
            with skipped_log.open("w", encoding="utf-8") as f:
                json.dump(skipped, f, indent=2)
            summary["skipped_log"] = str(skipped_log)

    if not entries:
        msg = f"No patches extracted from {source_dir}"
        if skipped:
            msg += f" ({len(skipped)} JP2 files failed to decode)"
        raise RuntimeError(msg)

    if manifest_path is None:
        manifest_path = default_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if entries:
        fieldnames = list(entries[0].keys())
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
        summary["manifest_csv"] = str(manifest_path)

    return entries, summary


def _scale_to_uint8(patch: np.ndarray) -> np.ndarray:
    """Robust min-max scale single-band patch to uint8 for PIL."""
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if finite.size == 0:
        return np.zeros(patch.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((patch - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


class HiRISEPost2025PatchDataset(Dataset):
    """Flat patch dataset; __getitem__ returns (tensor, dummy_label=0)."""

    def __init__(
        self,
        entries: Sequence[Dict[str, Any]],
        transform: T.Compose | None = None,
    ):
        self.entries = list(entries)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        patch_file = Path(entry.get("patch_file") or entry["cache_path"])
        if patch_file.is_file() and patch_file.suffix == ".npy" and "_y" in patch_file.name:
            patch = np.load(patch_file)
        else:
            patches = np.load(entry["cache_path"], mmap_mode="r")
            patch = patches[entry["patch_idx"]]
        patch = np.asarray(patch, dtype=np.float32)
        gray = _scale_to_uint8(patch)
        rgb = np.stack([gray, gray, gray], axis=-1)
        pil = Image.fromarray(rgb, mode="RGB")
        if self.transform is not None:
            tensor = self.transform(pil)
        else:
            tensor = T.ToTensor()(pil)
        return tensor, 0


def split_patch_entries(
    entries: Sequence[Dict[str, Any]],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Image-level train/eval split over a flat patch index."""
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
        "train_images": len(train_stems),
        "eval_images": len(eval_stems),
        "train_patches": len(train_entries),
        "eval_patches": len(eval_entries),
        "num_train": len(train_entries),
        "num_eval": len(eval_entries),
    }
    return train_entries, eval_entries, info


def build_post2025_datasets(
    entries: Sequence[Dict[str, Any]],
    transform: T.Compose | None = None,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    train_entries, eval_entries, info = split_patch_entries(
        entries, train_fraction=train_fraction, seed=seed
    )
    train_ds = HiRISEPost2025PatchDataset(train_entries, transform=transform)
    eval_ds = HiRISEPost2025PatchDataset(eval_entries, transform=transform)
    split_info = {
        "dataset": "hirise_post2025",
        "has_labels": False,
        "train_fraction": train_fraction,
        "seed": seed,
        **info,
    }
    return train_ds, eval_ds, split_info


def build_post2025_splits(
    source_dir: Path | None = None,
    cache_dir: Path | None = None,
    transform: T.Compose | None = None,
    train_fraction: float = 0.8,
    seed: int = 42,
    force_reslice: bool = False,
) -> Tuple[Dataset, Dataset, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Image-level 80/20 split of patches (all patches from a JP2 stay in one split).
    """
    source_dir = source_dir or default_post2025_dir()
    cache_dir = cache_dir or default_patch_cache_dir()

    entries, summary = build_patch_index(
        source_dir,
        cache_dir,
        force_reslice=force_reslice,
    )
    if not entries:
        raise RuntimeError(f"No valid patches extracted from {source_dir}")

    train_ds, eval_ds, split_info = build_post2025_datasets(
        entries,
        transform=transform,
        train_fraction=train_fraction,
        seed=seed,
    )
    split_info = {**summary, **split_info}
    return train_ds, eval_ds, split_info, entries


def validate_post2025_paths(
    source_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> Dict[str, Any]:
    source_dir = source_dir or default_post2025_dir()
    cache_dir = cache_dir or default_patch_cache_dir()
    jp2_files = list_jp2_files(source_dir)
    return {
        "dataset": "hirise_post2025",
        "source_dir": str(source_dir),
        "cache_dir": str(cache_dir),
        "n_jp2_files": len(jp2_files),
        "note": "Unsupervised JP2 patches; probe/F1 disabled.",
    }
