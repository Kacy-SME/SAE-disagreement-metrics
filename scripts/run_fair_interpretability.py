#!/usr/bin/env python3
"""Run Priority-1 interpretability metrics on all fair ablation runs and merge into saebench_fair.csv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe")


def fair_csv_path(version: str) -> Path:
    stem = "saebench_fair_v2" if version == "v2" else "saebench_fair"
    return PROJECT_ROOT / "results" / "ablations" / f"{stem}.csv"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interpretability metrics for fair ablation grid")
    parser.add_argument(
        "--metrics",
        default="all",
        help="causal, mono, tcav, or all (default: all)",
    )
    parser.add_argument("--backbone", action="append", default=None)
    parser.add_argument("--layer-depth", action="append", default=None)
    parser.add_argument("--sae-arch", action="append", default=None)
    parser.add_argument(
        "--fair-version",
        choices=("v1", "v2"),
        default="v1",
        help="Merge into saebench_fair.csv (v1) or saebench_fair_v2.csv (v2)",
    )
    parser.add_argument(
        "--cache-activations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache official probe ViT activations (default: on; shared per backbone+layer)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have mono/tcav (or other requested metrics) in saebench_metrics.json",
    )
    args = parser.parse_args()

    fair_csv = fair_csv_path(args.fair_version)
    cache_flag = "--cache-activations" if args.cache_activations else "--no-cache-activations"
    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "compute_saebench.py"),
        "--metrics-only",
        "--metrics",
        args.metrics,
        "--probe-landforms-only",
        "--no-probe-supplementary",
        cache_flag,
        "--results-dir",
        str(PROJECT_ROOT / "results" / "ablations"),
        "--data-dir",
        str(PROJECT_ROOT / "results"),
        "--merge-fair-csv",
        str(fair_csv),
        "--dataset",
        "hirise_v3_2",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.backbone:
        for bb in args.backbone:
            cmd.extend(["--backbone", bb])
    if args.layer_depth:
        for ld in args.layer_depth:
            cmd.extend(["--layer-depth", ld])
    if args.sae_arch:
        for sa in args.sae_arch:
            cmd.extend(["--sae-arch", sa])

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
