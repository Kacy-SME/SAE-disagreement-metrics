#!/usr/bin/env python3
"""
Generate metric histograms from fair-protocol CSVs (no re-training required).

Outputs under figures/metric_histograms/:
  - per_metric/<metric>_all_backbones.png — one histogram per metric, all backbones overlaid
  - per_backbone/<backbone>_all_metrics.png — all numeric metrics for one backbone
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metric_histograms import generate_metric_histograms  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "results" / "ablations" / "saebench_fair_v2.csv"
DEFAULT_PAPER = PROJECT_ROOT / "results" / "ablations" / "saebench_fair_paper_v2.csv"
OUTPUT_ROOT = PROJECT_ROOT / "figures" / "metric_histograms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--paper-csv", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fair-version", choices=("v1", "v2"), default="v2")
    return parser.parse_args()


def resolve_default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input_csv != DEFAULT_INPUT or args.fair_version == "v1":
        input_csv = args.input_csv
    else:
        stem = "saebench_fair_v2" if args.fair_version == "v2" else "saebench_fair"
        input_csv = PROJECT_ROOT / "results" / "ablations" / f"{stem}.csv"
    if args.paper_csv != DEFAULT_PAPER or args.fair_version == "v1":
        paper_csv = args.paper_csv
    else:
        stem = "saebench_fair_v2" if args.fair_version == "v2" else "saebench_fair"
        paper_csv = PROJECT_ROOT / "results" / "ablations" / f"{stem}_paper.csv"
    return input_csv, paper_csv


def main() -> None:
    args = parse_args()
    input_csv, paper_csv = resolve_default_paths(args)
    counts = generate_metric_histograms(
        input_csv,
        args.output_dir,
        paper_csv=paper_csv if paper_csv.is_file() else None,
        bins=args.bins,
        alpha=args.alpha,
        dpi=args.dpi,
    )
    print(
        f"Input: {input_csv} ({counts['n_metrics']} numeric metrics)\n"
        f"Wrote {counts['per_metric']} per-metric plots -> {args.output_dir / 'per_metric'}\n"
        f"Wrote {counts['per_backbone']} per-backbone plots -> {args.output_dir / 'per_backbone'}"
    )


if __name__ == "__main__":
    main()
