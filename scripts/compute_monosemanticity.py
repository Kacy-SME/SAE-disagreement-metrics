#!/usr/bin/env python3
"""
Compute monosemanticity proxy scores for all completed SAE runs.

Paper reference:
  https://transformer-circuits.pub/2023/monosemantic-features/

Metrics computed from saved feature_activation_matrix.npy + HiRISE labels:
  - feature density (per latent + run aggregates)
  - label purity on top-20 firing images (vision proxy for activation interpretability)
  - activation-interval label consistency (11 bins, paper-style)

For full paper automated interpretability (LLM explanation + activation prediction),
see scripts/README_monosemanticity.md or SAEBench — requires API keys and is
not run by default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.monosemanticity import score_run_directory  # noqa: E402


def discover_runs(results_root: Path) -> list[Path]:
    runs = []
    for matrix_path in sorted(results_root.rglob("feature_activation_matrix.npy")):
        runs.append(matrix_path.parent)
    return runs


def parse_run_path(run_dir: Path, results_root: Path) -> dict:
    rel = run_dir.relative_to(results_root)
    parts = rel.parts
    if len(parts) != 3:
        return {"backbone": "", "layer_depth": "", "sae_arch": ""}
    return {
        "backbone": parts[0],
        "layer_depth": parts[1],
        "sae_arch": parts[2],
    }


def main() -> None:
    results_root = PROJECT_ROOT / "results"
    split_manifest = results_root / "data_split_manifest.json"
    readable_cache = results_root / "hirise_readable_cache.json"
    exp_cfg_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    with exp_cfg_path.open(encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    n_eval = int(exp_cfg["data"]["eval_images"])

    rows = []
    for run_dir in discover_runs(results_root):
        meta = parse_run_path(run_dir, results_root)
        scored = score_run_directory(
            run_dir,
            split_manifest,
            readable_cache,
            n_eval_images=n_eval,
        )
        if scored is None:
            continue
        row = {**meta, **{k: v for k, v in scored.items() if not isinstance(v, (list, dict))}}
        rows.append(row)
        print(
            f"{meta['backbone']}/{meta['layer_depth']}/{meta['sae_arch']}: "
            f"purity_top20={row.get('mean_label_purity_top20', 0):.3f} "
            f"interval_consistency={row.get('mean_interval_label_consistency', float('nan')):.3f} "
            f"median_density={row.get('median_feature_density', 0):.4f} "
            f"alive={row.get('fraction_alive', 0):.3f}"
        )
        # Per-run JSON (without large arrays)
        out_json = run_dir / "monosemanticity_proxies.json"
        skip = {"feature_density", "label_purity_top20", "interval_label_consistency"}
        payload = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in scored.items()
            if k not in skip
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if not rows:
        print("No feature_activation_matrix.npy files found under results/")
        return

    out_csv = results_root / "monosemanticity_scores.csv"
    df = pd.DataFrame(rows)
    if (results_root / "summary_table.csv").is_file():
        summary = pd.read_csv(results_root / "summary_table.csv")
        key = ["backbone", "layer_depth", "sae_arch"]
        df = df.merge(summary, on=key, how="left", suffixes=("", "_summary"))
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} runs)")


if __name__ == "__main__":
    main()
