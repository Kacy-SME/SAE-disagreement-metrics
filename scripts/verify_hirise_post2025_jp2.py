#!/usr/bin/env python3
"""
Verify JP2 files decode (catches partial Google Drive / interrupted downloads).

Matches the Colab download layout: hirise_post_may2025/*.JP2 from HiRISE PDS RDR.

Example:
  python scripts/verify_hirise_post2025_jp2.py
  python scripts/verify_hirise_post2025_jp2.py --source-dir "G:\\My Drive\\hirise_post_may2025"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hirise_post2025 import (  # noqa: E402
    default_post2025_dir,
    list_jp2_files,
    read_jp2_band,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "diagnostics" / "hirise_post2025_jp2_verify.json",
    )
    parser.add_argument("--min-size-mb", type=float, default=1.0, help="Flag tiny files as bad")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir or default_post2025_dir()
    files = list_jp2_files(source_dir)

    from tqdm import tqdm

    ok: list[dict] = []
    bad: list[dict] = []

    print(f"Decoding {len(files)} JP2s (1–3+ min each for large scenes)...", flush=True)
    for path in tqdm(files, desc="Verify JP2", unit="file"):
        size_mb = path.stat().st_size / 1e6
        if size_mb < args.min_size_mb:
            bad.append(
                {"file": path.name, "size_mb": round(size_mb, 2), "error": "file too small"}
            )
            tqdm.write(f"BAD  {path.name} ({size_mb:.1f} MB) — too small")
            continue
        try:
            arr = read_jp2_band(path)
            ok.append(
                {
                    "file": path.name,
                    "size_mb": round(size_mb, 2),
                    "shape": list(arr.shape),
                }
            )
            tqdm.write(f"OK   {path.name} ({size_mb:.1f} MB) shape={arr.shape}")
        except Exception as exc:
            bad.append({"file": path.name, "size_mb": round(size_mb, 2), "error": str(exc)})
            tqdm.write(f"BAD  {path.name} ({size_mb:.1f} MB) — {exc}")

    report = {
        "source_dir": str(source_dir),
        "n_total": len(files),
        "n_ok": len(ok),
        "n_bad": len(bad),
        "ok": ok,
        "bad": bad,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{len(ok)} OK / {len(bad)} BAD / {len(files)} total")
    print(f"Report: {args.output}")
    if bad:
        print("\nRe-download bad files from Colab (delete local copy first):")
        for row in bad:
            stem = row["file"].replace("_RED.JP2", "")
            print(f"  https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ (search {stem})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
