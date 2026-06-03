#!/usr/bin/env python3
"""Update eval_metrics.json and summary_table.csv using saved H_* fields (no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones.registry import list_experiment_combinations, load_backbone_configs
from src.eval.metrics import compute_loss_recovered  # noqa: E402

LAYER_INDEX = {
    (c["backbone"], c["layer_depth"], c["sae_arch"]): c["layer_index"]
    for c in list_experiment_combinations(configs=load_backbone_configs())
}


def refresh_eval_file(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    h_with = metrics["H_with_sae"]
    h_zero = metrics["H_zero_ablate"]
    h_orig = metrics["H_original"]
    lr, valid = compute_loss_recovered(h_with, h_zero, h_orig)
    mse = (h_zero - h_with) / h_zero if h_zero > 0 else 0.0
    metrics["loss_recovered"] = lr
    metrics["loss_recovered_valid"] = valid
    metrics["mse_reduction_vs_zero"] = mse
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    results = PROJECT_ROOT / "results"
    rows = []
    for path in sorted(results.rglob("eval_metrics.json")):
        parts = path.relative_to(results).parts
        if len(parts) < 4:
            continue
        backbone, layer_depth, sae_arch = parts[0], parts[1], parts[2]
        m = refresh_eval_file(path)
        layer_index = LAYER_INDEX.get((backbone, layer_depth, sae_arch), "")
        rows.append(
            {
                "backbone": backbone,
                "layer_depth": layer_depth,
                "layer_index": layer_index,
                "sae_arch": sae_arch,
                "loss_recovered": m["loss_recovered"],
                "loss_recovered_valid": m["loss_recovered_valid"],
                "mse_reduction_vs_zero": m["mse_reduction_vs_zero"],
                "mean_L0": m.get("mean_L0"),
                "dead_fraction": m.get("dead_fraction"),
                "absorption_proxy": m.get("absorption_proxy"),
            }
        )
        print(
            f"{backbone}/{layer_depth}/{sae_arch}: "
            f"loss_recovered={m['loss_recovered']:.4f} "
            f"valid={m['loss_recovered_valid']} "
            f"mse_vs_zero={m['mse_reduction_vs_zero']:.3f}"
        )

    # Merge with existing summary for failed runs (no eval_metrics.json)
    summary_path = results / "summary_table.csv"
    if summary_path.is_file():
        old = pd.read_csv(summary_path)
        refreshed = pd.DataFrame(rows)
        key = ["backbone", "layer_depth", "sae_arch"]
        merged = old[~old[key].apply(tuple, axis=1).isin(refreshed[key].apply(tuple, axis=1))]
        out = pd.concat([refreshed, merged], ignore_index=True)
    else:
        out = pd.DataFrame(rows)
    out.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
