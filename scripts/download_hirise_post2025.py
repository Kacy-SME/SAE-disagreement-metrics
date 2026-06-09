#!/usr/bin/env python3
"""
Download post-May 2025 HiRISE RED RDR JP2s (port of Colab Untitled18.ipynb).

Source: https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/
Default save: G:\\My Drive\\hirise_post_may2025 (or HIRISE_POST2025_DIR)

Skips existing files unless --force. Use verify_hirise_post2025_jp2.py after download.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hirise_post2025 import default_post2025_dir  # noqa: E402

BASE = "https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/"

TARGET_ORBS = [
    "ORB_087600_087699",
    "ORB_088000_088099",
    "ORB_088400_088499",
    "ORB_088800_088899",
    "ORB_089200_089299",
    "ORB_089600_089699",
    "ORB_090000_090099",
    "ORB_090400_090499",
    "ORB_090800_090899",
    "ORB_091400_091499",
    "ORB_091900_091999",
    "ORB_092300_092399",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--max-per-orb", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--orb", action="append", default=None, help="Limit to orbit folder(s)")
    return parser.parse_args()


def list_observations(orb_folder: str) -> list[str]:
    url = BASE + orb_folder + "/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return [
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].startswith("ESP_")
    ]


def get_rdr_jp2(obs_url: str) -> list[str]:
    r = requests.get(obs_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return [
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].endswith("_RED.JP2")
    ]


def download_file(url: str, out_path: Path) -> float:
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (50 * 1024 * 1024) < len(chunk):
                        pct = 100.0 * downloaded / total
                        print(f"    ... {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", flush=True)
    return out_path.stat().st_size / 1e6


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir or default_post2025_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    orbs = args.orb or TARGET_ORBS

    for orb in orbs:
        print(f"\n{orb}")
        try:
            obs_list = list_observations(orb)[: args.max_per_orb]
        except Exception as exc:
            print(f"  Failed to list: {exc}")
            continue

        for obs in obs_list:
            obs_url = BASE + orb + "/" + obs
            try:
                files = get_rdr_jp2(obs_url)
                if not files:
                    print(f"  {obs}: no RED JP2 found")
                    continue
                fname = files[0]
                out_path = save_dir / fname
                if out_path.exists() and not args.force:
                    print(f"  {fname}: already exists, skipping")
                    continue
                file_url = obs_url + fname
                print(f"  Downloading {fname}...")
                size_mb = download_file(file_url, out_path)
                print(f"  OK {size_mb:.1f} MB")
            except Exception as exc:
                print(f"  {obs}: error — {exc}")

    print(f"\nDone. Files in {save_dir}")
    print("Run: python scripts/verify_hirise_post2025_jp2.py")


if __name__ == "__main__":
    main()
