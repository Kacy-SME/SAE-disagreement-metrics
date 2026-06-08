#!/usr/bin/env python3
"""
Extract missing HiRISE JPGs from hirise-map-proj-v3_2.zip into images/.

Use --landforms-only to pull official train+test landforms first (~65 MB, reaches
311 test images). Use --images-dir on a drive with free space if Google Drive
cache on C: is full.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from tqdm import tqdm

DEFAULT_ROOT = Path(r"G:\My Drive\metrics&models\mars_data\hirise_v3_2")
ZIP_NAME = "hirise-map-proj-v3_2.zip"


def landform_relpaths(labels_root: Path) -> set[str]:
    split_file = labels_root / "labels-map-proj-v3_2_train_val_test.txt"
    rels: set[str] = set()
    with split_file.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            rel, cls, split = parts[0], parts[1], parts[2]
            if cls != "0" and split in ("train", "val", "test"):
                rels.add(rel)
    return rels


def members_to_extract(
    zf: zipfile.ZipFile,
    images_dir: Path,
    landforms_only: bool,
    labels_root: Path,
) -> list[zipfile.ZipInfo]:
    allowed = landform_relpaths(labels_root) if landforms_only else None
    out: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        if not info.filename.lower().endswith(".jpg"):
            continue
        if "__MACOSX" in info.filename or "/._" in info.filename:
            continue
        name = Path(info.filename).name
        if allowed is not None and name not in allowed:
            continue
        if (images_dir / name).is_file():
            continue
        out.append(info)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract missing HiRISE images from zip")
    parser.add_argument("--hirise-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Output folder (default: hirise-root/images). Use D:\\... if G: Drive cache fills C:.",
    )
    parser.add_argument(
        "--landforms-only",
        action="store_true",
        help="Official train+val+test landforms (classes 1-7) — 311 test + supplementary val+test eval",
    )
    parser.add_argument(
        "--all-missing",
        action="store_true",
        help="Extract every labeled JPG still missing from zip (larger)",
    )
    args = parser.parse_args()

    if not args.landforms_only and not args.all_missing:
        args.landforms_only = True

    root = args.hirise_root.resolve()
    images_dir = (args.images_dir or root / "images").resolve()
    zip_path = root / ZIP_NAME
    images_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        todo = members_to_extract(zf, images_dir, args.landforms_only, root)
        total_bytes = sum(i.file_size for i in todo)
        print(f"images_dir: {images_dir}")
        print(f"files to extract: {len(todo)} (~{total_bytes / 1e6:.1f} MB)")
        if not todo:
            print("Nothing to do.")
            return

        for info in tqdm(todo, desc="Extract", unit="file"):
            name = Path(info.filename).name
            target = images_dir / name
            if target.is_file():
                continue
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

    on_disk = len(list(images_dir.glob("*.jpg")))
    print(f"Done. JPG count in {images_dir}: {on_disk}")


if __name__ == "__main__":
    main()
