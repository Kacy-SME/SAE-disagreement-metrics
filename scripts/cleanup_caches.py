#!/usr/bin/env python3
"""
Report (and optionally delete) large regenerable caches.

Dry-run by default. Use --delete to actually remove files.

Example:
  python scripts/cleanup_caches.py
  python scripts/cleanup_caches.py --delete --targets post2025_c_slices,eval_patch_caches
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

# Regenerable caches — safe to delete if you can re-run extract/slice.
TARGETS: dict[str, list[Path]] = {
    "post2025_c_patches": [
        RESULTS / "cache" / "hirise_post2025" / "patches",
        RESULTS / "cache" / "hirise_post2025" / "slices",
    ],
    "post2025_d": [
        Path(r"D:\hirise_post2025_cache"),
    ],
    "probe_activation_cache": [
        RESULTS / "probe_activation_cache",
    ],
    "eval_patch_per_run": [],  # filled by glob below
    "spatial_heatmaps": [],  # filled by glob below
}


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def collect_eval_patch_caches() -> list[Path]:
    return sorted(RESULTS.rglob("eval_patch_activations.pt"))


def collect_spatial_heatmaps() -> list[Path]:
    return sorted(RESULTS.rglob("spatial_heatmaps"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        default="all",
        help="Comma-separated keys or 'all'. Keys: " + ", ".join(TARGETS),
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete (default: dry-run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TARGETS["eval_patch_per_run"] = collect_eval_patch_caches()
    TARGETS["spatial_heatmaps"] = collect_spatial_heatmaps()

    keys = list(TARGETS) if args.targets == "all" else [k.strip() for k in args.targets.split(",")]
    total = 0

    print("Regenerable caches (NOT sae_weights.pt — keep those unless retraining):\n")
    for key in keys:
        if key not in TARGETS:
            print(f"Unknown target: {key}")
            continue
        paths = TARGETS[key]
        group_bytes = 0
        print(f"[{key}]")
        for p in paths:
            if not p.exists():
                print(f"  (missing) {p}")
                continue
            sz = dir_size(p)
            group_bytes += sz
            print(f"  {fmt_gb(sz):>8}  {p}")
        total += group_bytes
        if args.delete and paths:
            for p in paths:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            print(f"  -> deleted")
        print()

    print(f"Total selected: {fmt_gb(total)}")
    if not args.delete:
        print("\nDry-run only. Re-run with --delete to remove.")


if __name__ == "__main__":
    main()
