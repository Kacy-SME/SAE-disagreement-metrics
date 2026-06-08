#!/usr/bin/env python3
"""
Compute extra SAE metrics from cached probe activations + checkpoints (no ViT re-extract).

Metrics: dec_ortho, mono_ms, geo_hier, spatial, ood
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_saebench import (  # noqa: E402
    _load_probe_activation_cache,
    _shared_probe_cache_paths,
    discover_runs,
    filter_runs,
    merge_rows_into_fair_csv,
    parse_run_path,
    row_for_csv,
)
from src.backbones.registry import load_backbone_configs  # noqa: E402
from src.eval.fair_extra_metrics import (  # noqa: E402
    compute_extra_metrics,
    parse_extra_metrics_arg,
)
from src.eval.saebench_core import load_sae_for_eval  # noqa: E402

FAIR_CSV = PROJECT_ROOT / "results" / "ablations" / "saebench_fair.csv"
DATA_DIR = PROJECT_ROOT / "results"
RESULTS_ROOT = PROJECT_ROOT / "results" / "ablations"


def load_test_cache(
    data_dir: Path,
    backbone: str,
    layer_depth: str,
) -> tuple[torch.Tensor, list[str]] | None:
    """Load mmap-backed official test probe activations from shared cache."""
    train_path, eval_path = _shared_probe_cache_paths(
        data_dir, backbone, layer_depth, ["train"], ["test"]
    )
    if not eval_path.is_file():
        return None
    if train_path.is_file():
        loaded = _load_probe_activation_cache(train_path, eval_path)
    else:
        try:
            eval_payload = torch.load(
                eval_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            tqdm.write(f"[WARN] corrupt eval cache {eval_path}: {exc}")
            return None
        loaded = (
            None,
            None,
            0,
            eval_payload["activations"],
            eval_payload["labels"],
            int(eval_payload["n_images"]),
            {},
        )
    if loaded is None:
        return None
    return loaded[3], loaded[4]


def add_ood_gaps(fair_csv: Path) -> None:
    if not fair_csv.is_file():
        return
    df = pd.read_csv(fair_csv)
    if "sparse_probe_f1_sae_latents" in df.columns and "sparse_probe_supp_f1_sae_latents" in df.columns:
        df["ood_gap"] = df["sparse_probe_f1_sae_latents"] - df["sparse_probe_supp_f1_sae_latents"]
    if "sparse_probe_f1_backbone" in df.columns and "sparse_probe_supp_f1_backbone" in df.columns:
        df["ood_gap_backbone"] = df["sparse_probe_f1_backbone"] - df["sparse_probe_supp_f1_backbone"]
    df.to_csv(fair_csv, index=False)


def extra_metrics_complete(run_dir: Path, metrics: set[str]) -> bool:
    path = run_dir / "saebench_metrics.json"
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    need = {
        "dec_ortho": "dec_ortho_mean",
        "mono_ms": "mono_ms_mean",
        "geo_hier": "geo_hier_mean",
        "spatial": "spatial_cons_mean",
    }
    for m in metrics:
        if m == "ood":
            continue
        col = need.get(m)
        if col and col not in row:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra fair-protocol SAE metrics (cached activations)")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--fair-csv", default=str(FAIR_CSV))
    parser.add_argument(
        "--metrics",
        default="dec_ortho,mono_ms,geo_hier",
        help="dec_ortho, mono_ms, geo_hier, spatial, ood, all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--backbone", action="append", default=None)
    parser.add_argument("--layer-depth", action="append", default=None)
    parser.add_argument("--sae-arch", action="append", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--encode-chunk", type=int, default=512)
    args = parser.parse_args()

    metrics = parse_extra_metrics_arg(args.metrics)
    if not metrics:
        parser.error("No metrics selected")

    results_root = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    fair_csv = Path(args.fair_csv)

    with Path(args.config).open(encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    backbone_cfgs = load_backbone_configs()
    device = args.device or exp_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    runs = filter_runs(
        discover_runs(results_root),
        results_root,
        args.backbone,
        args.layer_depth,
        args.sae_arch,
    )

    compute_metrics = metrics - {"ood"}
    print(
        f"Runs: {len(runs)} | metrics={','.join(sorted(metrics))} | device={device}",
        flush=True,
    )

    rows: list[dict] = []
    for run_dir in tqdm(runs, desc="ExtraMetrics", unit="run"):
        meta = parse_run_path(run_dir, results_root)
        tag = f"{meta['backbone']}/{meta['layer_depth']}/{meta['sae_arch']}"
        if args.skip_existing and extra_metrics_complete(run_dir, compute_metrics):
            tqdm.write(f"{tag}: skip (extra metrics present)")
            continue

        row = {**meta, "saebench_status": "extra_metrics"}
        try:
            weights = run_dir / "sae_weights.pt"
            if not weights.is_file():
                row["saebench_status"] = "skip_no_weights"
                rows.append(row)
                continue

            if compute_metrics:
                cached = load_test_cache(data_dir, meta["backbone"], meta["layer_depth"])
                if cached is None:
                    raise FileNotFoundError(
                        f"Missing test cache for {meta['backbone']}/{meta['layer_depth']} "
                        f"(expected under {data_dir / 'probe_activation_cache'})"
                    )
                test_acts, test_labels = cached
                sae, _, norm_spec = load_sae_for_eval(weights, device)
                bb_cfg = backbone_cfgs[meta["backbone"]]
                row.update(
                    compute_extra_metrics(
                        sae,
                        test_acts,
                        test_labels,
                        norm_spec,
                        device,
                        bb_cfg,
                        run_dir,
                        compute_metrics,
                        encode_chunk=args.encode_chunk,
                    )
                )
                del sae, test_acts
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
                row["saebench_status"] = "extra:" + "+".join(sorted(compute_metrics))

            tqdm.write(f"{tag}: {row.get('saebench_status')}")
        except Exception as exc:
            row["saebench_status"] = f"error: {exc}"
            tqdm.write(f"{tag}: ERROR {exc}")
            traceback.print_exc()

        rows.append(row)
        out_json = run_dir / "saebench_metrics.json"
        if out_json.is_file():
            existing = json.loads(out_json.read_text(encoding="utf-8"))
            existing.update(row)
            row = existing
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, default=float)

    if rows and compute_metrics:
        merge_rows_into_fair_csv(fair_csv, rows)
        print(f"Merged extra metrics into {fair_csv}", flush=True)

    if "ood" in metrics:
        add_ood_gaps(fair_csv)
        print(f"Added ood_gap columns to {fair_csv}", flush=True)


if __name__ == "__main__":
    main()
