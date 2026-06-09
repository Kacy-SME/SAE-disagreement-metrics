#!/usr/bin/env python3
"""
Pre-slice HiRISE post-May 2025 JP2 images into 224x224 patches.

Writes:
  results/cache/hirise_post2025/patches/<stem>.npy   # float32 stack for training
  results/cache/hirise_post2025/patches/<stem>.json  # grid metadata
  results/cache/hirise_post2025/slices/<stem>/*.png  # viewable RGB slices
  results/cache/hirise_post2025/patch_manifest.csv   # flat index

Example:
  python scripts/cache_hirise_post2025_patches.py
  python scripts/cache_hirise_post2025_patches.py --source-dir "G:\\My Drive\\hirise_post_may2025"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hirise_post2025 import (  # noqa: E402
    build_patch_index,
    default_manifest_path,
    default_patch_cache_dir,
    default_post2025_dir,
    default_slices_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--slices-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Re-slice even if .npy exists")
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG export (keep .npy + manifest only)",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail on corrupt/undecodable JP2 files (default: skip and log)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir or default_post2025_dir()
    cache_dir = args.cache_dir or default_patch_cache_dir()
    slices_dir = None if args.no_png else (args.slices_dir or default_slices_dir())
    manifest = args.manifest or default_manifest_path()

    skipped_log = cache_dir.parent / "skipped_hirise_post2025_jp2.json"
    entries, summary = build_patch_index(
        source_dir=source_dir,
        cache_dir=cache_dir,
        force_reslice=args.force,
        slices_dir=slices_dir,
        save_png_slices=not args.no_png,
        manifest_path=manifest,
        strict=args.strict_data,
        skipped_log=skipped_log,
    )

    summary_path = cache_dir.parent / "slice_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Source:    {source_dir}")
    print(f"Images:    {summary['n_images']}")
    print(f"Patches:   {summary['n_patches']}")
    print(f"NPY cache: {cache_dir}")
    if not args.no_png:
        print(f"PNG slices:{slices_dir}")
    print(f"Manifest:  {summary.get('manifest_csv', manifest)}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
