#!/usr/bin/env python3
"""
Compute SAEBench-aligned metrics for HiRISE / ViT SAE runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones.registry import create_backbone, load_backbone_configs  # noqa: E402
from src.data.hirise import (  # noqa: E402
    HiRISEDataset,
    build_hirise_splits,
    filter_readable_official_samples,
    load_official_split_table,
    load_readable_cache,
    official_split_class_counts,
    save_official_probe_manifest,
)
from src.eval.metrics import absorption_proxy  # noqa: E402
from src.eval.monosemanticity import load_eval_labels  # noqa: E402
from src.eval.saebench_core import compute_core_metrics, load_sae_for_eval  # noqa: E402
from src.eval.saebench_sparse_probing import compute_hirise_sparse_probing  # noqa: E402
from src.extract.activations import activation_dataloader, extract_patch_activations  # noqa: E402


def discover_runs(results_root: Path) -> list[Path]:
    return sorted(p.parent for p in results_root.rglob("sae_weights.pt"))


def parse_run_path(run_dir: Path, results_root: Path) -> dict:
    rel = run_dir.relative_to(results_root)
    parts = rel.parts
    return {
        "backbone": parts[0] if len(parts) > 0 else "",
        "layer_depth": parts[1] if len(parts) > 1 else "",
        "sae_arch": parts[2] if len(parts) > 2 else "",
    }


def load_layer_index(backbone: str, layer_depth: str, sae_arch: str) -> int:
    from src.backbones.registry import list_experiment_combinations, load_backbone_configs

    for c in list_experiment_combinations(configs=load_backbone_configs()):
        if (
            c["backbone"] == backbone
            and c["layer_depth"] == layer_depth
            and c["sae_arch"] == sae_arch
        ):
            return int(c["layer_index"])
    return 0


def proxy_metrics_from_disk(
    run_dir: Path, split_manifest: Path, readable_cache: Path
) -> dict:
    out = {}
    eval_path = run_dir / "eval_metrics.json"
    if eval_path.is_file():
        m = json.loads(eval_path.read_text(encoding="utf-8"))
        out["proxy_loss_recovered"] = m.get("loss_recovered")
        out["proxy_mse_reduction_vs_zero"] = m.get("mse_reduction_vs_zero")
        out["proxy_mean_l0"] = m.get("mean_L0")
        out["proxy_dead_fraction"] = m.get("dead_fraction")
        out["proxy_absorption_fraction"] = m.get("absorption_proxy")

    matrix_path = run_dir / "feature_activation_matrix.npy"
    if matrix_path.is_file():
        import numpy as np

        matrix = np.load(matrix_path)
        labels = load_eval_labels(split_manifest, readable_cache, matrix.shape[1])
        out["proxy_absorption_fraction"] = absorption_proxy(matrix)
        if labels:
            from src.eval.monosemanticity import label_purity_topk

            purities = label_purity_topk(matrix, labels, top_k=20)
            alive = (matrix > 0).any(axis=1)
            if alive.any():
                out["proxy_label_purity_top20"] = float(purities[alive].mean())
    return out


def append_log(results_root: Path, message: str) -> None:
    log_path = results_root / "saebench_run.log"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def row_for_csv(row: dict) -> dict:
    return {k: v for k, v in row.items() if not isinstance(v, (dict, list))}


def save_scores_csv(results_root: Path, rows: list[dict]) -> None:
    out_csv = results_root / "saebench_scores.csv"
    new_df = pd.DataFrame([row_for_csv(r) for r in rows])
    if out_csv.is_file() and len(new_df) > 0:
        key = ["backbone", "layer_depth", "sae_arch"]
        old_df = pd.read_csv(out_csv)
        kept = old_df[
            ~old_df[key].apply(tuple, axis=1).isin(new_df[key].apply(tuple, axis=1))
        ]
        new_df = pd.concat([new_df, kept], ignore_index=True)
    new_df = new_df.sort_values(["backbone", "layer_depth", "sae_arch"]).reset_index(
        drop=True
    )
    new_df.to_csv(out_csv, index=False)


@torch.no_grad()
def extract_eval_activations(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    split_manifest: Path,
    readable_cache: Path,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    cache_acts: bool,
) -> tuple[torch.Tensor, list[str], int]:
    meta = parse_run_path(run_dir, results_root)
    cache_path = run_dir / "eval_patch_activations.pt"
    n_eval = min(int(exp_cfg["data"]["eval_images"]), 500)

    if cache_acts and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        labels = load_eval_labels(split_manifest, readable_cache, payload["n_images"])
        return payload["activations"], labels, int(payload["n_images"])

    layer_index = load_layer_index(
        meta["backbone"], meta["layer_depth"], meta["sae_arch"]
    )
    with split_manifest.open(encoding="utf-8") as f:
        split = json.load(f)
    images_dir = split["images_dir"]
    labels_file = split["labels_file"]

    cached = load_readable_cache(readable_cache)
    if cached is None:
        raise FileNotFoundError(f"Missing readable cache: {readable_cache}")
    samples = [
        {
            "path": Path(images_dir) / rel,
            "rel_path": rel,
            "label": cached["labels_by_path"].get(rel, ""),
        }
        for rel in sorted(cached.get("readable_rel_paths", []))
    ]

    backbone = create_backbone(
        meta["backbone"],
        device=device,
        configs=backbone_cfgs,
        allow_prithvi_rgb_proxy=allow_prithvi_rgb_proxy,
    )
    transform = backbone.get_transform()
    _, eval_set, _, _ = build_hirise_splits(
        images_dir=images_dir,
        labels_file=labels_file,
        transform=transform,
        train_fraction=float(exp_cfg["data"]["train_fraction"]),
        seed=int(exp_cfg.get("seed", 42)),
        readable_samples=samples,
    )
    n_eval = min(int(exp_cfg["data"]["eval_images"]), len(eval_set))
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
        desc=f"SAEBench {meta['backbone']}/{meta['layer_depth']}",
    )
    labels = load_eval_labels(split_manifest, readable_cache, n_eval)

    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    acts_cpu = acts.cpu().float()
    if cache_acts:
        torch.save(
            {"activations": acts_cpu, "n_images": n_eval},
            cache_path,
        )
    return acts_cpu, labels, n_eval


def _readable_sample_list(readable_cache: Path, images_dir: str) -> list[dict]:
    cached = load_readable_cache(readable_cache)
    if cached is None:
        raise FileNotFoundError(f"Missing readable cache: {readable_cache}")
    return [
        {
            "path": Path(images_dir) / rel,
            "rel_path": rel,
            "label": cached["labels_by_path"].get(rel, ""),
        }
        for rel in sorted(cached.get("readable_rel_paths", []))
    ]


@torch.no_grad()
def extract_official_probe_activations(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    readable_cache: Path,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    cache_acts: bool,
    train_splits: list[str],
) -> tuple[torch.Tensor, list[str], int, torch.Tensor, list[str], int, dict]:
    """
    ViT forward on official HiRISE train/test landform splits (classes 1–7).

    Train split fits the linear probe; test split is the 311-image creator holdout.
    """
    meta = parse_run_path(run_dir, results_root)
    train_cache = run_dir / "official_probe_train_activations.pt"
    test_cache = run_dir / "official_probe_test_activations.pt"

    if cache_acts and train_cache.is_file() and test_cache.is_file():
        train_payload = torch.load(train_cache, map_location="cpu", weights_only=False)
        test_payload = torch.load(test_cache, map_location="cpu", weights_only=False)
        counts = test_payload.get("test_class_counts", {})
        return (
            train_payload["activations"],
            train_payload["labels"],
            int(train_payload["n_images"]),
            test_payload["activations"],
            test_payload["labels"],
            int(test_payload["n_images"]),
            counts,
        )

    layer_index = load_layer_index(
        meta["backbone"], meta["layer_depth"], meta["sae_arch"]
    )
    with (results_root / "data_split_manifest.json").open(encoding="utf-8") as f:
        split = json.load(f)
    images_dir = split["images_dir"]
    labels_file = split["labels_file"]

    readable_set = set(load_readable_cache(readable_cache).get("readable_rel_paths", []))
    split_rows = load_official_split_table(labels_file)
    train_samples = filter_readable_official_samples(
        split_rows, readable_set, train_splits, landforms_only=True
    )
    test_samples = filter_readable_official_samples(
        split_rows, readable_set, ["test"], landforms_only=True
    )
    manifest_path = save_official_probe_manifest(results_root, train_samples, test_samples)
    test_class_counts = official_split_class_counts(test_samples)
    if len(test_samples) < 280:
        tqdm.write(
            f"[WARN] Official test landforms readable: {len(test_samples)}/311 "
            f"(see {manifest_path.name}; download missing HiRISE v3.2 test images)."
        )

    backbone = create_backbone(
        meta["backbone"],
        device=device,
        configs=backbone_cfgs,
        allow_prithvi_rgb_proxy=allow_prithvi_rgb_proxy,
    )
    transform = backbone.get_transform()
    batch_size = int(exp_cfg["data"]["activation_batch_size"])
    num_workers = int(exp_cfg["data"].get("dataloader_workers", 0))

    def forward_samples(samples: list[dict], desc: str) -> torch.Tensor:
        if not samples:
            raise RuntimeError(f"{desc}: no readable official-split images")
        dataset = HiRISEDataset(
            images_dir=images_dir,
            labels_file=labels_file,
            transform=transform,
            samples=[
                {
                    "path": Path(images_dir) / s["rel_path"],
                    "rel_path": s["rel_path"],
                    "label": s["label"],
                }
                for s in samples
            ],
        )
        loader = activation_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return extract_patch_activations(
            backbone=backbone,
            dataloader=loader,
            layer_index=layer_index,
            device=device,
            desc=desc,
        )

    train_acts = forward_samples(
        train_samples,
        f"Probe-train {meta['backbone']}/{meta['layer_depth']}",
    ).cpu().float()
    test_acts = forward_samples(
        test_samples,
        f"Probe-test {meta['backbone']}/{meta['layer_depth']}",
    ).cpu().float()
    train_labels = [s["label"] for s in train_samples]
    test_labels = [s["label"] for s in test_samples]

    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    if cache_acts:
        torch.save(
            {"activations": train_acts, "labels": train_labels, "n_images": len(train_labels)},
            train_cache,
        )
        torch.save(
            {
                "activations": test_acts,
                "labels": test_labels,
                "n_images": len(test_labels),
                "test_class_counts": test_class_counts,
            },
            test_cache,
        )

    return (
        train_acts,
        train_labels,
        len(train_labels),
        test_acts,
        test_labels,
        len(test_labels),
        test_class_counts,
    )


def process_run(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    split_manifest: Path,
    readable_cache: Path,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    do_extract: bool,
    cache_acts: bool,
    core_batch_size: int,
    landforms_only: bool = False,
    probe_official_test: bool = False,
    probe_train_splits: list[str] | None = None,
) -> dict:
    meta = parse_run_path(run_dir, results_root)
    row = {**meta, "saebench_status": "proxy_only"}
    row.update(proxy_metrics_from_disk(run_dir, split_manifest, readable_cache))

    if not do_extract:
        return row

    weights = run_dir / "sae_weights.pt"
    if not weights.is_file():
        row["saebench_status"] = "skip_no_weights"
        return row

    use_prithvi_proxy = allow_prithvi_rgb_proxy or meta["backbone"] == "prithvi_eo_2"
    acts, labels, n_images = extract_eval_activations(
        run_dir,
        results_root,
        exp_cfg,
        backbone_cfgs,
        split_manifest,
        readable_cache,
        device,
        use_prithvi_proxy,
        cache_acts=cache_acts,
    )
    sae, _, norm_scalar = load_sae_for_eval(weights, device)
    core = compute_core_metrics(
        sae, acts, norm_scalar=norm_scalar, batch_size=core_batch_size
    )
    row.update(core)
    if landforms_only and probe_official_test:
        (
            train_acts,
            train_labels,
            n_train,
            test_acts,
            test_labels,
            n_test,
            test_counts,
        ) = extract_official_probe_activations(
            run_dir,
            results_root,
            exp_cfg,
            backbone_cfgs,
            readable_cache,
            device,
            allow_prithvi_rgb_proxy,
            cache_acts,
            train_splits=probe_train_splits or ["train"],
        )
        sparse = compute_hirise_sparse_probing(
            sae,
            norm_scalar=norm_scalar,
            landforms_only=True,
            train_backbone_acts=train_acts,
            train_labels=train_labels,
            test_backbone_acts=test_acts,
            test_labels=test_labels,
            official_split=True,
            test_class_counts=test_counts,
        )
        row["saebench_status"] = "core+sparse+proxy+official_test311"
    else:
        sparse = compute_hirise_sparse_probing(
            sae,
            acts,
            n_images,
            labels,
            norm_scalar=norm_scalar,
            landforms_only=landforms_only,
        )
        row["saebench_status"] = (
            "core+sparse+proxy+landforms7" if landforms_only else "core+sparse+proxy"
        )
    row.update(sparse)

    del sae, acts
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="SAEBench-aligned metrics for HiRISE SAE runs")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-prithvi-rgb-proxy",
        action="store_true",
        help="Enable Prithvi RGB proxy (auto-on for prithvi_eo_2 runs)",
    )
    parser.add_argument(
        "--cache-activations",
        action="store_true",
        help="Save eval_patch_activations.pt per run to skip ViT re-forward on retry",
    )
    parser.add_argument(
        "--core-batch-size",
        type=int,
        default=1024,
        help="Batch size for core metrics (lower if CUDA OOM)",
    )
    parser.add_argument("--backbone", action="append", default=None)
    parser.add_argument("--layer-depth", action="append", default=None)
    parser.add_argument("--sae-arch", action="append", default=None)
    parser.add_argument(
        "--probe-landforms-only",
        action="store_true",
        help=(
            "Sparse probing on HiRISE classes 1–7 only (exclude class 0 other). "
            "Recommended for interpretable landform detection claims."
        ),
    )
    parser.add_argument(
        "--probe-official-test",
        action="store_true",
        help=(
            "Evaluate sparse probes on the official HiRISE v3.2 test split "
            "(311 non-augmented landform images); train probe on official train split."
        ),
    )
    parser.add_argument(
        "--probe-adhoc-eval",
        action="store_true",
        help=(
            "Legacy probe eval: random 80/20 holdout on the first N ad-hoc eval images (~87 landforms)."
        ),
    )
    parser.add_argument(
        "--probe-train-splits",
        default="train",
        help="Comma-separated official splits for probe training (default: train).",
    )
    args = parser.parse_args()

    if args.probe_landforms_only and not args.probe_adhoc_eval:
        args.probe_official_test = True

    if not args.proxy_only and not args.extract:
        print("Using --proxy-only. Pass --extract for Core + Sparse Probing.")
        args.proxy_only = True

    results_root = Path(args.results_dir)
    split_manifest = results_root / "data_split_manifest.json"
    readable_cache = results_root / "hirise_readable_cache.json"
    with Path(args.config).open(encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    backbone_cfgs = load_backbone_configs()

    device = args.device or exp_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU (slow).")
        device = "cpu"

    runs = discover_runs(results_root)
    if args.backbone:
        runs = [r for r in runs if r.parts[-3] in args.backbone]
    if args.layer_depth:
        runs = [r for r in runs if r.parts[-2] in args.layer_depth]
    if args.sae_arch:
        runs = [r for r in runs if r.parts[-1] in args.sae_arch]

    if args.probe_landforms_only:
        if args.probe_official_test and not args.probe_adhoc_eval:
            probe_mode = "official test (311 landforms, classes 1–7)"
        else:
            probe_mode = "adhoc holdout (~87 landforms)"
    else:
        probe_mode = "all 8 classes (incl. other)"
    train_splits = [s.strip() for s in args.probe_train_splits.split(",") if s.strip()]
    print(
        f"Runs: {len(runs)} | device={device} | extract={args.extract} | probe={probe_mode}",
        flush=True,
    )
    rows: list[dict] = []

    for run_dir in tqdm(runs, desc="SAEBench", unit="run"):
        meta = parse_run_path(run_dir, results_root)
        tag = f"{meta['backbone']}/{meta['layer_depth']}/{meta['sae_arch']}"
        try:
            row = process_run(
                run_dir,
                results_root,
                exp_cfg,
                backbone_cfgs,
                split_manifest,
                readable_cache,
                device,
                args.allow_prithvi_rgb_proxy,
                do_extract=args.extract,
                cache_acts=args.cache_activations,
                core_batch_size=args.core_batch_size,
                landforms_only=args.probe_landforms_only,
                probe_official_test=args.probe_official_test and not args.probe_adhoc_eval,
                probe_train_splits=train_splits,
            )
            tqdm.write(
                f"{tag}: {row.get('saebench_status')} "
                f"EV={row.get('core_explained_variance', '—')} "
                f"F1_bb={row.get('sparse_probe_f1_backbone', '—')} "
                f"F1_sae={row.get('sparse_probe_f1_sae_latents', '—')} "
                f"(n_test={row.get('sparse_probe_num_images', '—')} "
                f"n_train={row.get('sparse_probe_num_train_images', '—')})"
            )
        except Exception as exc:
            row = {**meta, "saebench_status": f"error: {exc}"}
            err_text = traceback.format_exc()
            append_log(results_root, f"FAILED {tag}\n{err_text}")
            tqdm.write(f"{tag}: ERROR {exc}")

        rows.append(row)
        out_json = run_dir / "saebench_metrics.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, default=float)
        save_scores_csv(results_root, rows)

    print(f"\nWrote {results_root / 'saebench_scores.csv'} ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    main()
