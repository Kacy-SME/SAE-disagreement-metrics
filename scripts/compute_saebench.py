#!/usr/bin/env python3
"""
Compute SAEBench-aligned metrics for HiRISE / ViT SAE runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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


def resolve_data_paths(
    results_root: Path, data_dir: Path | None = None
) -> tuple[Path, Path]:
    """Locate split manifest (post-2025 training metadata)."""
    if data_dir is not None:
        base = data_dir
    elif (results_root / "data_split_manifest.json").is_file():
        base = results_root
    elif (PROJECT_ROOT / "results" / "data_split_manifest.json").is_file():
        base = PROJECT_ROOT / "results"
    else:
        base = results_root
    return base, base / "data_split_manifest.json"


def dataset_tag_columns() -> dict[str, str]:
    return {
        "sae_train_dataset": "post2025_hirise",
        "eval_dataset": EVAL_DATASET_NAME,
    }

from src.backbones.registry import create_backbone, load_backbone_configs  # noqa: E402
from src.data.marsbench import (  # noqa: E402
    EVAL_DATASET_NAME,
    MarsBenchProbeDataset,
    build_marsbench_probe_splits,
    default_marsbench_root,
)
from src.data.post_may2025 import (  # noqa: E402
    PostMay2025IndexedDataset,
    build_post_may2025_datasets,
    default_patch_cache_dir,
)
from src.eval.obs_div import (  # noqa: E402
    compute_observation_diversity,
    obs_div_summary_row,
    save_obs_div_per_latent_csv,
)
from src.eval.metrics import absorption_proxy  # noqa: E402
from src.eval.saebench_core import compute_core_metrics, load_sae_for_eval  # noqa: E402
from src.eval.saebench_interpretability import (  # noqa: E402
    compute_interpretability_metrics,
    parse_metrics_arg,
)
from src.eval.saebench_sparse_probing import compute_hirise_sparse_probing  # noqa: E402
from src.extract.activations import (  # noqa: E402
    activation_dataloader,
    extract_image_mean_activations,
    extract_patch_activations,
)


def discover_runs(results_root: Path) -> list[Path]:
    return sorted(p.parent for p in results_root.rglob("sae_weights.pt"))


def interpretability_complete(run_dir: Path, metrics: set[str]) -> bool:
    """True if saebench_metrics.json already has requested interpretability metrics."""
    path = run_dir / "saebench_metrics.json"
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    status = str(row.get("saebench_status", ""))
    if not status.startswith("interpretability:"):
        return False
    if "error" in status:
        return False
    need = {
        "mono": "mono_mean_purity_top20_official",
        "tcav": "tcav_mean_max_cos_per_class",
        "causal": "causal_mean_f1_drop",
    }
    for key in metrics:
        col = need.get(key)
        if col and col not in row:
            return False
    return True


def filter_runs(
    runs: list[Path],
    results_root: Path,
    backbones: list[str] | None,
    layer_depths: list[str] | None,
    sae_archs: list[str] | None,
) -> list[Path]:
    out: list[Path] = []
    for run_dir in runs:
        meta = parse_run_path(run_dir, results_root)
        if backbones and meta["backbone"] not in backbones:
            continue
        if layer_depths and meta["layer_depth"] not in layer_depths:
            continue
        if sae_archs and meta["sae_arch"] not in sae_archs:
            continue
        out.append(run_dir)
    return out


def parse_run_path(run_dir: Path, results_root: Path) -> dict:
    """
    Parse run metadata from results directory layout.

    Fair ablations use:
      {backbone}/{ablation_tag}/{backbone}/{layer_depth}/{sae_arch}/
    Legacy grid uses:
      {backbone}/{layer_depth}/{sae_arch}/
    """
    rel = run_dir.relative_to(results_root)
    parts = rel.parts
    if len(parts) >= 5 and parts[0] == parts[2]:
        return {
            "backbone": parts[0],
            "ablation_tag": parts[1],
            "layer_depth": parts[3],
            "sae_arch": parts[4],
        }
    return {
        "backbone": parts[0] if len(parts) > 0 else "",
        "ablation_tag": parts[1] if len(parts) > 2 else "",
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


def proxy_metrics_from_disk(run_dir: Path) -> dict:
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
        out["proxy_absorption_fraction"] = absorption_proxy(matrix)
    obs_div_path = run_dir / "obs_div_per_latent.csv"
    if obs_div_path.is_file():
        df = pd.read_csv(obs_div_path)
        if "obs_div" in df.columns and len(df):
            out["obs_div_mean"] = float(df["obs_div"].mean())
            out["obs_div_median"] = float(df["obs_div"].median())
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
def extract_marsbench_eval_activations(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    cache_acts: bool,
    marsbench_root: Path | None = None,
    max_eval_samples: int | None = None,
) -> tuple[torch.Tensor, list[str], int]:
    meta = parse_run_path(run_dir, results_root)
    cache_path = run_dir / "marsbench_eval_activations.pt"

    if cache_acts and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (
            payload["activations"],
            payload.get("labels", []),
            int(payload["n_images"]),
        )

    layer_index = load_layer_index(
        meta["backbone"], meta["layer_depth"], meta["sae_arch"]
    )
    backbone = create_backbone(
        meta["backbone"],
        device=device,
        configs=backbone_cfgs,
        allow_prithvi_rgb_proxy=allow_prithvi_rgb_proxy,
    )
    transform = backbone.get_transform()
    _train_ds, eval_ds, _summary = build_marsbench_probe_splits(
        transform=transform,
        root=marsbench_root or default_marsbench_root(),
        eval_splits=("test",),
        max_eval_samples=max_eval_samples,
    )
    n_eval = min(int(exp_cfg["data"]["eval_images"]), len(eval_ds))
    eval_subset = torch.utils.data.Subset(eval_ds, list(range(n_eval)))
    loader = activation_dataloader(
        eval_subset,
        batch_size=int(exp_cfg["data"]["activation_batch_size"]),
        num_workers=int(exp_cfg["data"].get("dataloader_workers", 0)),
    )
    acts = extract_image_mean_activations(
        backbone=backbone,
        dataloader=loader,
        layer_index=layer_index,
        device=device,
        desc=f"Mars-Bench eval {meta['backbone']}/{meta['layer_depth']}",
    )
    labels = eval_ds.labels[:n_eval]

    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    acts_cpu = acts.cpu().float()
    if cache_acts:
        torch.save(
            {"activations": acts_cpu, "labels": labels, "n_images": n_eval},
            cache_path,
        )
    return acts_cpu, labels, n_eval


def _probe_cache_tag(eval_splits: list[str]) -> str:
    return "_".join(s.lower() for s in eval_splits)


def _probe_act_cache_key(
    backbone: str,
    layer_depth: str,
    train_splits: list[str],
    eval_splits: list[str],
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    return (backbone, layer_depth, tuple(train_splits), tuple(eval_splits))


def _shared_probe_cache_paths(
    data_dir: Path,
    backbone: str,
    layer_depth: str,
    train_splits: list[str],
    eval_splits: list[str],
) -> tuple[Path, Path]:
    base = data_dir / "probe_activation_cache" / backbone / layer_depth
    train_tag = _probe_cache_tag(train_splits)
    eval_tag = _probe_cache_tag(eval_splits)
    return (
        base / f"train_{train_tag}_activations.pt",
        base / f"eval_{eval_tag}_activations.pt",
    )


def _run_dir_probe_cache_paths(
    run_dir: Path,
    eval_splits: list[str],
) -> tuple[Path, Path]:
    eval_tag = _probe_cache_tag(eval_splits)
    train_cache = run_dir / "official_probe_train_activations.pt"
    eval_cache = run_dir / f"official_probe_eval_{eval_tag}_activations.pt"
    legacy_test_cache = run_dir / "official_probe_test_activations.pt"
    if eval_splits == ["test"] and not eval_cache.is_file() and legacy_test_cache.is_file():
        eval_cache = legacy_test_cache
    return train_cache, eval_cache


def _load_probe_activation_cache(
    train_path: Path,
    eval_path: Path,
) -> tuple[torch.Tensor, list[str], int, torch.Tensor, list[str], int, dict] | None:
    min_bytes = 1_000_000  # truncated saves from Ctrl+C are ~24 KB
    try:
        if train_path.stat().st_size < min_bytes or eval_path.stat().st_size < min_bytes:
            raise ValueError("cache file too small (likely interrupted save)")
        train_payload = torch.load(
            train_path, map_location="cpu", mmap=True, weights_only=False
        )
        eval_payload = torch.load(eval_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        tqdm.write(f"[WARN] Ignoring corrupt probe cache {train_path.parent.name}: {exc}")
        for p in (train_path, eval_path):
            if p.is_file():
                p.unlink(missing_ok=True)
        return None
    counts = eval_payload.get("eval_class_counts") or eval_payload.get(
        "test_class_counts", {}
    )
    return (
        train_payload["activations"],
        train_payload["labels"],
        int(train_payload["n_images"]),
        eval_payload["activations"],
        eval_payload["labels"],
        int(eval_payload["n_images"]),
        counts,
    )


def _save_probe_activation_cache(
    train_path: Path,
    eval_path: Path,
    train_acts: torch.Tensor,
    train_labels: list[str],
    eval_acts: torch.Tensor,
    eval_labels: list[str],
    eval_splits: list[str],
    eval_class_counts: dict,
) -> None:
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = train_path.with_suffix(".pt.tmp")
    eval_tmp = eval_path.with_suffix(".pt.tmp")
    torch.save(
        {"activations": train_acts, "labels": train_labels, "n_images": len(train_labels)},
        train_tmp,
    )
    torch.save(
        {
            "activations": eval_acts,
            "labels": eval_labels,
            "n_images": len(eval_labels),
            "eval_splits": eval_splits,
            "eval_class_counts": eval_class_counts,
        },
        eval_tmp,
    )
    train_tmp.replace(train_path)
    eval_tmp.replace(eval_path)


def _find_sibling_run_probe_cache(
    results_root: Path,
    backbone: str,
    layer_depth: str,
    eval_splits: list[str],
) -> tuple[Path, Path] | None:
    """Find probe caches under fair ablation layout: {bb}/{tag}/{bb}/{layer}/{arch}/."""
    min_bytes = 1_000_000
    bb_dir = results_root / backbone
    if not bb_dir.is_dir():
        return None
    for ablation_dir in sorted(bb_dir.iterdir()):
        if not ablation_dir.is_dir():
            continue
        layer_dir = ablation_dir / backbone / layer_depth
        if not layer_dir.is_dir():
            continue
        for run_sub in sorted(layer_dir.iterdir()):
            if not run_sub.is_dir():
                continue
            train_p, eval_p = _run_dir_probe_cache_paths(run_sub, eval_splits)
            if (
                train_p.is_file()
                and eval_p.is_file()
                and train_p.stat().st_size >= min_bytes
                and eval_p.stat().st_size >= min_bytes
            ):
                return train_p, eval_p
    return None


def _persist_probe_activation_cache(
    cache_acts: bool,
    data_dir: Path,
    results_root: Path,
    run_dir: Path,
    meta: dict,
    train_splits: list[str],
    eval_splits: list[str],
    train_acts: torch.Tensor,
    train_labels: list[str],
    eval_acts: torch.Tensor,
    eval_labels: list[str],
    eval_class_counts: dict,
) -> None:
    if not cache_acts:
        return
    shared_train, shared_eval = _shared_probe_cache_paths(
        data_dir, meta["backbone"], meta["layer_depth"], train_splits, eval_splits
    )
    _save_probe_activation_cache(
        shared_train,
        shared_eval,
        train_acts,
        train_labels,
        eval_acts,
        eval_labels,
        eval_splits,
        eval_class_counts,
    )


def prefix_probe_metrics(sparse: dict, tag: str) -> dict:
    """Prefix sparse_probe_* keys (e.g. tag=supp -> sparse_probe_supp_f1_backbone)."""
    if not tag:
        return sparse
    out: dict = {}
    for key, val in sparse.items():
        if key.startswith("sparse_probe_"):
            out[f"sparse_probe_{tag}_{key[len('sparse_probe_'):]}"] = val
        else:
            out[key] = val
    return out


@torch.no_grad()
def extract_marsbench_probe_activations(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    cache_acts: bool,
    train_splits: list[str],
    eval_splits: list[str],
    data_dir: Path,
    marsbench_root: Path | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[torch.Tensor, list[str], int, torch.Tensor, list[str], int, dict]:
    """ViT forward on pooled Mars-Bench splits for sparse probing / interpretability."""
    meta = parse_run_path(run_dir, results_root)
    shared_train, shared_eval = _shared_probe_cache_paths(
        data_dir, meta["backbone"], meta["layer_depth"], train_splits, eval_splits
    )

    if cache_acts and shared_train.is_file() and shared_eval.is_file():
        result = _load_probe_activation_cache(shared_train, shared_eval)
        if result is not None:
            tqdm.write(
                f"[cache] {meta['backbone']}/{meta['layer_depth']} Mars-Bench probe activations"
            )
            return result

    layer_index = load_layer_index(
        meta["backbone"], meta["layer_depth"], meta["sae_arch"]
    )
    backbone = create_backbone(
        meta["backbone"],
        device=device,
        configs=backbone_cfgs,
        allow_prithvi_rgb_proxy=allow_prithvi_rgb_proxy,
    )
    transform = backbone.get_transform()
    root = marsbench_root or default_marsbench_root()
    train_ds, eval_ds, summary = build_marsbench_probe_splits(
        transform=transform,
        root=root,
        train_splits=tuple(train_splits),
        eval_splits=tuple(eval_splits),
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    batch_size = int(exp_cfg["data"]["activation_batch_size"])
    num_workers = int(exp_cfg["data"].get("dataloader_workers", 0))

    def forward_dataset(dataset: MarsBenchProbeDataset, desc: str) -> torch.Tensor:
        loader = activation_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return extract_image_mean_activations(
            backbone=backbone,
            dataloader=loader,
            layer_index=layer_index,
            device=device,
            desc=desc,
        )

    train_acts = forward_dataset(
        train_ds, f"Mars-Bench probe-train {meta['backbone']}/{meta['layer_depth']}"
    ).cpu().float()
    eval_acts = forward_dataset(
        eval_ds,
        f"Mars-Bench probe-eval {meta['backbone']}/{meta['layer_depth']}",
    ).cpu().float()
    train_labels = train_ds.labels
    eval_labels = eval_ds.labels
    eval_class_counts = summary.get("eval_class_counts", {})

    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    _persist_probe_activation_cache(
        cache_acts,
        data_dir,
        results_root,
        run_dir,
        meta,
        train_splits,
        eval_splits,
        train_acts,
        train_labels,
        eval_acts,
        eval_labels,
        eval_class_counts,
    )
    return (
        train_acts,
        train_labels,
        len(train_labels),
        eval_acts,
        eval_labels,
        len(eval_labels),
        eval_class_counts,
    )


def run_marsbench_sparse_probe(
    sae,
    norm_spec: dict,
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    cache_acts: bool,
    train_splits: list[str],
    eval_splits: list[str],
    data_dir: Path,
    marsbench_root: Path | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict:
    (
        train_acts,
        train_labels,
        _n_train,
        eval_acts,
        eval_labels,
        _n_eval,
        eval_counts,
    ) = extract_marsbench_probe_activations(
        run_dir,
        results_root,
        exp_cfg,
        backbone_cfgs,
        device,
        allow_prithvi_rgb_proxy,
        cache_acts,
        train_splits=train_splits,
        eval_splits=eval_splits,
        data_dir=data_dir,
        marsbench_root=marsbench_root,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    split_label = "marsbench_test" if eval_splits == ["test"] else "marsbench_val_test"
    return compute_hirise_sparse_probing(
        sae,
        norm_spec=norm_spec,
        landforms_only=False,
        train_backbone_acts=train_acts,
        train_labels=train_labels,
        test_backbone_acts=eval_acts,
        test_labels=eval_labels,
        official_split=True,
        test_class_counts=eval_counts,
    ) | {
        "sparse_probe_eval_splits": ",".join(eval_splits),
        "sparse_probe_split": split_label,
    }


def run_interpretability_for_run(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    cache_acts: bool,
    metrics: set[str],
    probe_train_splits: list[str],
    data_dir: Path,
    marsbench_root: Path | None = None,
) -> dict:
    weights = run_dir / "sae_weights.pt"
    if not weights.is_file():
        return {"saebench_status": "skip_no_weights"}
    sae, _, norm_spec = load_sae_for_eval(weights, device)
    sae.eval()
    (
        train_acts,
        train_labels,
        _,
        test_acts,
        test_labels,
        _,
        _,
    ) = extract_marsbench_probe_activations(
        run_dir,
        results_root,
        exp_cfg,
        backbone_cfgs,
        device,
        allow_prithvi_rgb_proxy,
        cache_acts,
        train_splits=probe_train_splits,
        eval_splits=["test"],
        data_dir=data_dir,
        marsbench_root=marsbench_root,
    )
    out = compute_interpretability_metrics(
        sae,
        train_acts,
        train_labels,
        test_acts,
        test_labels,
        norm_spec,
        device,
        run_dir,
        metrics,
    )
    del train_acts
    del sae, test_acts
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return out


def merge_rows_into_fair_csv(fair_csv: Path, rows: list[dict]) -> None:
    """Append interpretability columns to saebench_fair.csv by run key."""
    if not fair_csv.is_file() or not rows:
        return
    key = ["backbone", "layer_depth", "sae_arch"]
    fair = pd.read_csv(fair_csv)
    new_df = pd.DataFrame([row_for_csv(r) for r in rows])
    if not all(k in fair.columns for k in key):
        return
    fair_keys = fair[key].apply(tuple, axis=1)
    for _, row in new_df.iterrows():
        rk = tuple(row[k] for k in key)
        mask = fair_keys == rk
        if not mask.any():
            continue
        for col in row.index:
            if col not in key:
                fair.loc[mask, col] = row[col]
    fair.to_csv(fair_csv, index=False)


def compute_obs_div_for_run(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    post2025_cache_dir: Path | None = None,
) -> dict:
    weights = run_dir / "sae_weights.pt"
    if not weights.is_file():
        return {}
    meta = parse_run_path(run_dir, results_root)
    layer_index = load_layer_index(
        meta["backbone"], meta["layer_depth"], meta["sae_arch"]
    )
    sae, _, norm_spec = load_sae_for_eval(weights, device)
    backbone = create_backbone(
        meta["backbone"],
        device=device,
        configs=backbone_cfgs,
        allow_prithvi_rgb_proxy=allow_prithvi_rgb_proxy,
    )
    transform = backbone.get_transform()
    _train_ds, _eval_ds, _info, _te, eval_entries = build_post_may2025_datasets(
        cache_dir=post2025_cache_dir or default_patch_cache_dir(),
        transform=transform,
        train_fraction=float(exp_cfg["data"]["train_fraction"]),
        seed=int(exp_cfg.get("seed", 42)),
    )
    n_eval = min(int(exp_cfg["data"]["eval_images"]), len(eval_entries))
    eval_slice = eval_entries[:n_eval]
    eval_ds = PostMay2025IndexedDataset(eval_slice, transform=transform)
    loader = activation_dataloader(
        eval_ds,
        batch_size=int(exp_cfg["data"]["activation_batch_size"]),
        num_workers=int(exp_cfg["data"].get("dataloader_workers", 0)),
    )
    obs_metrics = compute_observation_diversity(
        sae=sae,
        backbone=backbone,
        dataloader=loader,
        obs_ids=eval_ds.obs_ids,
        layer_index=layer_index,
        norm_spec=norm_spec,
        device=device,
    )
    per_latent = obs_metrics.pop("obs_div_per_latent")
    save_obs_div_per_latent_csv(per_latent, run_dir / "obs_div_per_latent.csv")
    del sae, backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return obs_div_summary_row(obs_metrics)


def process_run(
    run_dir: Path,
    results_root: Path,
    exp_cfg: dict,
    backbone_cfgs: dict,
    device: str,
    allow_prithvi_rgb_proxy: bool,
    do_extract: bool,
    cache_acts: bool,
    core_batch_size: int,
    probe_official_test: bool = False,
    probe_supplementary: bool = False,
    probe_train_splits: list[str] | None = None,
    data_dir: Path | None = None,
    metrics: set[str] | None = None,
    metrics_only: bool = False,
    skip_probe: bool = False,
    marsbench_root: Path | None = None,
    post2025_cache_dir: Path | None = None,
    marsbench_max_train: int | None = None,
    marsbench_max_eval: int | None = None,
) -> dict:
    meta = parse_run_path(run_dir, results_root)
    row = {**meta, **dataset_tag_columns(), "saebench_status": "proxy_only"}
    _data_dir, _split_manifest = resolve_data_paths(results_root, data_dir)
    row.update(proxy_metrics_from_disk(run_dir))
    metrics = metrics or set()

    if metrics_only and metrics:
        row.update(
            run_interpretability_for_run(
                run_dir,
                results_root,
                exp_cfg,
                backbone_cfgs,
                device,
                allow_prithvi_rgb_proxy,
                cache_acts,
                metrics,
                probe_train_splits or ["train", "val"],
                _data_dir,
                marsbench_root=marsbench_root,
            )
        )
        row["saebench_status"] = "interpretability:" + "+".join(sorted(metrics))
        return row

    if not do_extract:
        return row

    weights = run_dir / "sae_weights.pt"
    if not weights.is_file():
        row["saebench_status"] = "skip_no_weights"
        return row

    sae, _, norm_spec = load_sae_for_eval(weights, device)
    use_prithvi_proxy = allow_prithvi_rgb_proxy or meta["backbone"] == "prithvi_eo_2"
    acts, labels, n_images = extract_marsbench_eval_activations(
        run_dir,
        results_root,
        exp_cfg,
        backbone_cfgs,
        device,
        use_prithvi_proxy,
        cache_acts=cache_acts,
        marsbench_root=marsbench_root,
        max_eval_samples=marsbench_max_eval,
    )
    core = compute_core_metrics(
        sae, acts, norm_spec=norm_spec, batch_size=core_batch_size
    )
    row.update(core)
    row.update(
        compute_obs_div_for_run(
            run_dir,
            results_root,
            exp_cfg,
            backbone_cfgs,
            device,
            allow_prithvi_rgb_proxy,
            post2025_cache_dir=post2025_cache_dir,
        )
    )
    if skip_probe:
        row["saebench_status"] = "core+obs_div+proxy (probe disabled)"
        row["probe_disabled"] = True
    elif probe_official_test:
        sparse = run_marsbench_sparse_probe(
            sae,
            norm_spec,
            run_dir,
            results_root,
            exp_cfg,
            backbone_cfgs,
            device,
            allow_prithvi_rgb_proxy,
            cache_acts,
            train_splits=probe_train_splits or ["train", "val"],
            eval_splits=["test"],
            data_dir=_data_dir,
            marsbench_root=marsbench_root,
            max_train_samples=marsbench_max_train,
            max_eval_samples=marsbench_max_eval,
        )
        row.update(sparse)
        status = "core+sparse+obs_div+marsbench_test"
        if probe_supplementary:
            supp = run_marsbench_sparse_probe(
                sae,
                norm_spec,
                run_dir,
                results_root,
                exp_cfg,
                backbone_cfgs,
                device,
                allow_prithvi_rgb_proxy,
                cache_acts,
                train_splits=probe_train_splits or ["train", "val"],
                eval_splits=["val", "test"],
                data_dir=_data_dir,
                marsbench_root=marsbench_root,
                max_train_samples=marsbench_max_train,
                max_eval_samples=marsbench_max_eval,
            )
            row.update(prefix_probe_metrics(supp, "supp"))
            status += "+supp_val_test"
        row["saebench_status"] = status
    else:
        sparse = compute_hirise_sparse_probing(
            sae,
            acts,
            n_images,
            labels,
            norm_spec=norm_spec,
            landforms_only=False,
        )
        row.update(sparse)
        row["saebench_status"] = "core+sparse+obs_div+marsbench"

    if metrics and probe_official_test and not skip_probe:
        row.update(
            run_interpretability_for_run(
                run_dir,
                results_root,
                exp_cfg,
                backbone_cfgs,
                device,
                allow_prithvi_rgb_proxy,
                cache_acts,
                metrics,
                probe_train_splits or ["train", "val"],
                _data_dir,
                marsbench_root=marsbench_root,
            )
        )
        row["saebench_status"] += "+interp:" + "+".join(sorted(metrics))

    del sae, acts
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="SAEBench-aligned metrics for HiRISE SAE runs")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--data-dir",
        default=None,
        help="HiRISE manifest/cache directory (default: auto from results/ or --results-dir)",
    )
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save activation .pt caches per run (default: on; pass --no-cache-activations to disable)",
    )
    parser.add_argument(
        "--marsbench-root",
        type=Path,
        default=None,
        help="Mars-Bench data root (default: MARS_BENCH_ROOT or Drive/Mars-Bench)",
    )
    parser.add_argument(
        "--post2025-cache-dir",
        type=Path,
        default=None,
        help="Post-2025 patch cache for obs_div (default: D:\\hirise_post2025_cache\\patches)",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip Mars-Bench sparse probe F1",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cap Mars-Bench probe/eval samples for fast local testing",
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
    parser.add_argument(
        "--probe-supplementary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With official landform probing: also eval on official val+test (~4050 images). "
            "Metrics are prefixed sparse_probe_supp_* in CSV/JSON."
        ),
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated interpretability metrics: causal, mono, tcav, all",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip core/sparse probing; only run --metrics groups (needs probe caches or ViT extract)",
    )
    parser.add_argument(
        "--merge-fair-csv",
        type=Path,
        default=None,
        help="Merge new metric columns into this CSV (e.g. results/ablations/saebench_fair.csv)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose saebench_metrics.json already has the requested interpretability metrics",
    )
    args = parser.parse_args()

    if args.probe_landforms_only and not args.probe_adhoc_eval:
        args.probe_official_test = True

    metrics = parse_metrics_arg(args.metrics)
    if args.metrics_only and not metrics:
        parser.error("--metrics-only requires --metrics causal,mono,tcav,all")

    if not args.proxy_only and not args.extract and not args.metrics_only:
        print("Using --proxy-only. Pass --extract for Core + Sparse Probing.")
        args.proxy_only = True

    results_root = Path(args.results_dir)
    data_dir_arg = Path(args.data_dir) if args.data_dir else None
    data_dir, _split_manifest = resolve_data_paths(results_root, data_dir_arg)
    with Path(args.config).open(encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    backbone_cfgs = load_backbone_configs()

    device = args.device or exp_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU (slow).")
        device = "cpu"

    runs = discover_runs(results_root)
    runs = filter_runs(
        runs,
        results_root,
        args.backbone,
        args.layer_depth,
        args.sae_arch,
    )

    if args.probe_official_test and not args.probe_adhoc_eval:
        probe_mode = "Mars-Bench test split"
        if args.probe_supplementary:
            probe_mode += " + supplementary val+test"
    else:
        probe_mode = "Mars-Bench pooled holdout"
    train_splits = [s.strip() for s in args.probe_train_splits.split(",") if s.strip()]
    metrics_bit = f" | metrics={','.join(sorted(metrics))}" if metrics else ""
    print(
        f"Runs: {len(runs)} | device={device} | extract={args.extract} | "
        f"metrics_only={args.metrics_only} | probe={probe_mode}{metrics_bit}",
        flush=True,
    )
    rows: list[dict] = []
    skip_probe = args.skip_probe
    marsbench_max_train = 2000 if args.smoke else None
    marsbench_max_eval = 200 if args.smoke else None
    if args.smoke:
        exp_cfg = dict(exp_cfg)
        exp_cfg["data"] = dict(exp_cfg["data"])
        exp_cfg["data"]["eval_images"] = min(int(exp_cfg["data"]["eval_images"]), 200)

    for run_dir in tqdm(runs, desc="SAEBench", unit="run"):
        meta = parse_run_path(run_dir, results_root)
        tag = f"{meta['backbone']}/{meta['layer_depth']}/{meta['sae_arch']}"
        if args.skip_existing and args.metrics_only and interpretability_complete(
            run_dir, metrics
        ):
            tqdm.write(f"{tag}: skip (interpretability already complete)")
            continue
        try:
            row = process_run(
                run_dir,
                results_root,
                exp_cfg,
                backbone_cfgs,
                device,
                args.allow_prithvi_rgb_proxy,
                do_extract=args.extract and not args.metrics_only,
                cache_acts=args.cache_activations,
                core_batch_size=args.core_batch_size,
                probe_official_test=args.probe_official_test and not args.probe_adhoc_eval,
                probe_supplementary=args.probe_supplementary and not args.probe_adhoc_eval,
                probe_train_splits=train_splits,
                data_dir=data_dir,
                metrics=metrics,
                metrics_only=args.metrics_only,
                skip_probe=skip_probe,
                marsbench_root=args.marsbench_root,
                post2025_cache_dir=args.post2025_cache_dir,
                marsbench_max_train=marsbench_max_train,
                marsbench_max_eval=marsbench_max_eval,
            )
            supp_n = row.get("sparse_probe_supp_num_images")
            supp_f1 = row.get("sparse_probe_supp_f1_sae_latents")
            supp_bit = (
                f" | supp n={supp_n} F1_sae={supp_f1}" if supp_n is not None else ""
            )
            tqdm.write(
                f"{tag}: {row.get('saebench_status')} "
                f"EV={row.get('core_explained_variance', '—')} "
                f"F1_bb={row.get('sparse_probe_f1_backbone', '—')} "
                f"F1_sae={row.get('sparse_probe_f1_sae_latents', '—')} "
                f"(n_test={row.get('sparse_probe_num_images', '—')} "
                f"n_train={row.get('sparse_probe_num_train_images', '—')})"
                f"{supp_bit}"
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

    if args.merge_fair_csv and rows:
        merge_rows_into_fair_csv(Path(args.merge_fair_csv), rows)
        print(f"Merged metrics into {args.merge_fair_csv}", flush=True)


if __name__ == "__main__":
    main()
