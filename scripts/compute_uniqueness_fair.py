#!/usr/bin/env python3
"""
Compute ECCU Uniqueness Score for fair-protocol paper winners on HiRISE v3.2
official 311-image test split.

Reads results/ablations/saebench_fair_paper.csv, loads cached SAE activation
matrices, updates UniquenessScorer on all test images, scores, compares to a
shuffled-label random baseline, and writes uniqueness_* columns back to the CSV.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.uniqueness import (  # noqa: E402
    UniquenessScorer,
    activations_image_major,
    beats_random,
    live_latent_mask,
    mean_uniqueness_live,
    random_baseline_mean,
)
from src.paths import RESULTS_DIR  # noqa: E402

PAPER_CSV = RESULTS_DIR / "ablations" / "saebench_fair_paper.csv"
DEFAULT_ALPHA = 0.5
DEFAULT_LIVE_EPS = 1e-8
EXPECTED_TEST_IMAGES = 311


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-csv", type=Path, default=PAPER_CSV)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--live-eps", type=float, default=DEFAULT_LIVE_EPS)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_dir_for_row(row: pd.Series, results_dir: Path) -> Path:
    return (
        results_dir
        / "ablations"
        / row["backbone"]
        / row["ablation_tag"]
        / row["backbone"]
        / row["layer_depth"]
        / row["sae_arch"]
    )


def activation_matrix_path(run_dir: Path) -> Path:
    return run_dir / "official_test_activation_matrix.npy"


def probe_test_cache_path(row: pd.Series, results_dir: Path) -> Path:
    return (
        results_dir
        / "probe_activation_cache"
        / row["backbone"]
        / row["layer_depth"]
        / "eval_test_activations.pt"
    )


def load_test_labels(cache_path: Path) -> list[str]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    labels = [str(x) for x in payload["labels"]]
    return labels


def compute_for_row(
    row: pd.Series,
    results_dir: Path,
    alpha: float,
    live_eps: float,
    random_seed: int,
) -> dict:
    run_dir = run_dir_for_row(row, results_dir)
    matrix_path = activation_matrix_path(run_dir)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing activation matrix {matrix_path}")

    cache_path = probe_test_cache_path(row, results_dir)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing probe test cache {cache_path}")

    matrix = np.load(matrix_path)
    labels = load_test_labels(cache_path)

    if matrix.ndim != 2:
        raise ValueError(f"{matrix_path}: expected 2D matrix, got {matrix.shape}")

    acts_img = activations_image_major(matrix)
    n_images = acts_img.shape[0]
    if n_images != len(labels):
        raise ValueError(
            f"{row['backbone']}: matrix images {n_images} != labels {len(labels)}"
        )
    if n_images != EXPECTED_TEST_IMAGES:
        warnings.warn(
            f"{row['backbone']}: expected {EXPECTED_TEST_IMAGES} test images, got {n_images}"
        )

    result = mean_uniqueness_live(
        activations=acts_img,
        class_labels=labels,
        alpha=alpha,
        activation_threshold=0.0,
        live_eps=live_eps,
    )
    shuffled_mean = random_baseline_mean(
        activations=acts_img,
        class_labels=labels,
        alpha=alpha,
        activation_threshold=0.0,
        live_eps=live_eps,
        seed=random_seed,
    )
    beats = beats_random(result.mean_uniqueness, shuffled_mean)

    return {
        "uniqueness_mean": result.mean_uniqueness,
        "uniqueness_alpha": alpha,
        "uniqueness_beats_random": beats,
        "uniqueness_shuffled_mean": shuffled_mean,
        "uniqueness_n_live": result.n_live,
        "uniqueness_matrix_path": str(matrix_path),
    }


def print_summary(df: pd.DataFrame) -> None:
    cols = ["backbone", "uniqueness_mean", "uniqueness_beats_random"]
    print("\nUniqueness summary (official 311-image test split)")
    print("-" * 56)
    print(f"{'backbone':<22} | {'uniqueness_mean':>16} | {'beats_random':>12}")
    print("-" * 56)
    for _, row in df.iterrows():
        mean_u = row.get("uniqueness_mean", float("nan"))
        beats = row.get("uniqueness_beats_random", False)
        mean_str = f"{mean_u:.6f}" if pd.notna(mean_u) else "nan"
        beats_str = str(bool(beats))
        print(f"{row['backbone']:<22} | {mean_str:>16} | {beats_str:>12}")
    print("-" * 56)


def main() -> None:
    args = parse_args()
    if not args.paper_csv.is_file():
        raise FileNotFoundError(f"Missing paper CSV {args.paper_csv}")

    df = pd.read_csv(args.paper_csv)
    required = {"backbone", "layer_depth", "sae_arch", "ablation_tag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Paper CSV missing columns: {sorted(missing)}")

    rows_out = []
    for _, row in df.iterrows():
        stats = compute_for_row(
            row,
            args.results_dir,
            alpha=args.alpha,
            live_eps=args.live_eps,
            random_seed=args.random_seed,
        )
        if not stats["uniqueness_beats_random"]:
            warnings.warn(
                f"{row['backbone']}: real uniqueness {stats['uniqueness_mean']:.6f} "
                f"<= shuffled {stats['uniqueness_shuffled_mean']:.6f}"
            )
        merged = {**row.to_dict(), **stats}
        rows_out.append(merged)
        print(
            f"{row['backbone']}: mean={stats['uniqueness_mean']:.6f} "
            f"shuffled={stats['uniqueness_shuffled_mean']:.6f} "
            f"live={stats['uniqueness_n_live']} beats_random={stats['uniqueness_beats_random']}"
        )

    out_df = pd.DataFrame(rows_out)
    for col in ("uniqueness_shuffled_mean", "uniqueness_n_live", "uniqueness_matrix_path"):
        if col in out_df.columns:
            out_df = out_df.drop(columns=[col])

    print_summary(out_df)

    if args.dry_run:
        print("\n(dry-run: CSV not written)")
        return

    out_df.to_csv(args.paper_csv, index=False)
    print(f"\nWrote {args.paper_csv}")


if __name__ == "__main__":
    main()
