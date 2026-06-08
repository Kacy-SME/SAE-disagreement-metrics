#!/usr/bin/env python3
"""
Compare loss_recovered with decode(0) zero-ablation vs CLS-zero patch proxy.

For ViTs with a CLS token, sae.decode(zeros) can look artificially strong because
patch activations share a global CLS component that latent zeroing does not remove.
This script zeros the CLS token before block 0, re-extracts patch activations, and
uses MSE(cls_zero_patches, normal_patches) as H_zero for an alternate loss_recovered.

Output: results/diagnostics/loss_recovered_cls_zero.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiments import activation_dataloader  # noqa: E402
from src.backbones.registry import (  # noqa: E402
    create_backbone,
    list_experiment_combinations,
    load_backbone_configs,
)
from src.data.hirise import build_hirise_splits, load_readable_samples_from_cache  # noqa: E402
from src.eval.metrics import evaluate_sae_on_activations  # noqa: E402
from src.extract.activations import extract_patch_activations  # noqa: E402
from src.paths import CONFIGS_DIR, DEFAULT_PATHS, RESULTS_DIR  # noqa: E402
from src.sae.activation_norm import norm_spec_for_eval  # noqa: E402
from src.sae.trainer import load_sae_checkpoint  # noqa: E402

PAPER_CSV = RESULTS_DIR / "ablations" / "saebench_fair_paper.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-csv", type=Path, default=PAPER_CSV)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "experiment.yaml")
    parser.add_argument("--hirise-images-dir", default=r"D:\hirise_v3_2\images")
    parser.add_argument(
        "--hirise-labels-file",
        default=DEFAULT_PATHS["hirise_labels_file"],
    )
    parser.add_argument("--eval-images", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", action="append", default=None)
    return parser.parse_args()


def run_dir_for_row(row: dict, results_dir: Path) -> Path:
    return (
        results_dir
        / "ablations"
        / row["backbone"]
        / row["ablation_tag"]
        / row["backbone"]
        / row["layer_depth"]
        / row["sae_arch"]
    )


def layer_index_for(backbone: str, layer_depth: str, sae_arch: str, cfgs: dict) -> int:
    for combo in list_experiment_combinations(
        backbone_names=[backbone],
        layer_depths=[layer_depth],
        sae_archs=[sae_arch],
        configs=cfgs,
    ):
        return int(combo["layer_index"])
    raise KeyError(f"No layer index for {backbone}/{layer_depth}/{sae_arch}")


def diagnose_run(
    row: dict,
    args: argparse.Namespace,
    exp_cfg: dict,
    backbone_cfgs: dict,
    readable_samples: list[dict],
) -> dict:
    run_dir = run_dir_for_row(row, args.results_dir)
    weights = run_dir / "sae_weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing {weights}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    sae, payload = load_sae_checkpoint(weights, device)
    norm_spec = norm_spec_for_eval(payload)
    backbone_name = row["backbone"]
    layer_depth = row["layer_depth"]
    layer_index = layer_index_for(
        backbone_name, layer_depth, row["sae_arch"], backbone_cfgs
    )

    backbone = create_backbone(backbone_name, device=device, configs=backbone_cfgs)
    transform = backbone.get_transform()
    _, eval_set, _, _ = build_hirise_splits(
        images_dir=args.hirise_images_dir,
        labels_file=args.hirise_labels_file,
        transform=transform,
        train_fraction=float(exp_cfg["data"]["train_fraction"]),
        seed=int(exp_cfg.get("seed", 42)),
        readable_samples=readable_samples,
    )
    n_eval = min(int(args.eval_images), len(eval_set))
    eval_subset = torch.utils.data.Subset(eval_set, list(range(n_eval)))
    loader = activation_dataloader(
        eval_subset,
        batch_size=int(exp_cfg["data"]["activation_batch_size"]),
        num_workers=int(exp_cfg["data"].get("dataloader_workers", 0)),
    )

    acts = extract_patch_activations(
        backbone=backbone,
        dataloader=loader,
        layer_index=layer_index,
        device=device,
        desc=f"{backbone_name}/{layer_depth} normal",
    )
    has_cls = int(backbone.spec.num_prefix_tokens) > 0
    acts_cls_zero = None
    if has_cls:
        acts_cls_zero = extract_patch_activations(
            backbone=backbone,
            dataloader=loader,
            layer_index=layer_index,
            device=device,
            desc=f"{backbone_name}/{layer_depth} cls_zero",
            zero_cls_token=True,
        )

    metrics = evaluate_sae_on_activations(
        sae=sae,
        activations=acts,
        norm_spec=norm_spec,
        device=device,
        activations_cls_zero=acts_cls_zero,
    )

    saved = {}
    eval_path = run_dir / "eval_metrics.json"
    if eval_path.is_file():
        saved = json.loads(eval_path.read_text(encoding="utf-8"))

    out = {
        "backbone": backbone_name,
        "layer_depth": layer_depth,
        "sae_arch": row["sae_arch"],
        "ablation_tag": row["ablation_tag"],
        "num_prefix_tokens": int(backbone.spec.num_prefix_tokens),
        "has_cls": has_cls,
        "loss_recovered_decode_zero": metrics["loss_recovered"],
        "loss_recovered_decode_zero_valid": metrics["loss_recovered_valid"],
        "mse_reduction_vs_zero": metrics["mse_reduction_vs_zero"],
        "H_with_sae": metrics["H_with_sae"],
        "H_zero_decode": metrics["H_zero_ablate"],
        "H_original": metrics["H_original"],
        "saved_loss_recovered": saved.get("loss_recovered"),
        "loss_recovered_cls_proxy": metrics.get("loss_recovered_cls_proxy"),
        "loss_recovered_cls_proxy_valid": metrics.get("loss_recovered_cls_proxy_valid"),
        "mse_reduction_vs_cls_proxy": metrics.get("mse_reduction_vs_cls_proxy"),
        "H_zero_cls_proxy": metrics.get("H_zero_cls_proxy"),
    }
    del backbone, sae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main() -> None:
    args = parse_args()
    exp_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    backbone_cfgs = load_backbone_configs()
    paper = pd.read_csv(args.paper_csv)
    if args.backbone:
        paper = paper[paper["backbone"].isin(args.backbone)]

    cache_path = args.results_dir / "hirise_readable_cache.json"
    readable_samples, _, _ = load_readable_samples_from_cache(
        cache_path,
        args.hirise_images_dir,
        args.hirise_labels_file,
    )
    if readable_samples is None:
        raise SystemExit(f"Missing readable cache: {cache_path}")

    rows: list[dict] = []
    for _, row in paper.iterrows():
        label = f"{row['backbone']}/{row['layer_depth']}/{row['sae_arch']}"
        try:
            result = diagnose_run(row.to_dict(), args, exp_cfg, backbone_cfgs, readable_samples)
            rows.append(result)
            if result["has_cls"]:
                tqdm.write(
                    f"{label}: decode_zero LR={result['loss_recovered_decode_zero']:.3f} "
                    f"cls_proxy LR={result['loss_recovered_cls_proxy']:.3f} "
                    f"H_zero_decode={result['H_zero_decode']:.4g} "
                    f"H_zero_cls={result['H_zero_cls_proxy']:.4g}"
                )
            else:
                tqdm.write(
                    f"{label}: no CLS (patch-only) decode_zero LR="
                    f"{result['loss_recovered_decode_zero']:.3f}"
                )
        except Exception as exc:
            tqdm.write(f"[ERROR] {label}: {exc}")

    out_dir = args.results_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loss_recovered_cls_zero.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(rows)} rows)", flush=True)
    if rows:
        show = [
            "backbone",
            "num_prefix_tokens",
            "loss_recovered_decode_zero",
            "loss_recovered_cls_proxy",
            "H_zero_decode",
            "H_zero_cls_proxy",
        ]
        print(pd.DataFrame(rows)[show].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
