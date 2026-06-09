#!/usr/bin/env python3
"""Multi-backbone SAE experiment runner."""

from __future__ import annotations

print("SAE-Experiments: starting (importing libraries)...", flush=True)

import argparse
import gc
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones.registry import (  # noqa: E402
    create_backbone,
    list_experiment_combinations,
    load_backbone_configs,
)
from src.data.post_may2025 import (  # noqa: E402
    PostMay2025IndexedDataset,
    build_post_may2025_datasets,
    default_patch_cache_dir,
    validate_post_may2025_cache,
)
from src.eval.metrics import (  # noqa: E402
    absorption_proxy,
    compute_feature_activation_matrix,
    evaluate_sae_on_activations,
)
from src.extract.activations import (  # noqa: E402
    ActivationBuffer,
    activation_dataloader,
    extract_patch_activations,
    stream_patch_activations_to_buffer,
)
from src.paths import CONFIGS_DIR, DEFAULT_PATHS, RESULTS_DIR, resolve_hf_home  # noqa: E402
from src.sae.activation_norm import (  # noqa: E402
    EXPERIMENTAL_PREPROCESS_MODES,
    fit_activation_norm,
    norm_spec_for_eval,
    sae_input_dim,
)
from src.sae.trainer import (  # noqa: E402
    build_sae,
    load_sae_checkpoint,
    save_sae_checkpoint,
    train_sae,
    verify_checkpoint_norm_consistency,
)

print("SAE-Experiments: libraries loaded.", flush=True)

RECOMMENDED_PYTHON = Path(r"c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe")


def check_runtime_environment() -> None:
    """Fail fast with a clear message when not using the project venv."""
    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency (timm). This usually means you ran base Anaconda "
            f"'python' instead of the Orbital_ViT venv.\n\n"
            f"  Recommended:\n"
            f'    & "{RECOMMENDED_PYTHON}" -u run_experiments.py ...\n\n'
            f"  Current interpreter: {sys.executable}\n"
            f"  Original error: {exc}"
        ) from exc
    if RECOMMENDED_PYTHON.is_file():
        current = Path(sys.executable).resolve()
        expected = RECOMMENDED_PYTHON.resolve()
        if current != expected:
            print(
                f"Note: using {current}\n"
                f"      (trained runs used {expected})",
                flush=True,
            )


class IndexedSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        return img, idx


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_hf_cache(hf_home: str | None = None) -> str:
    path = hf_home or resolve_hf_home()
    os.environ["HF_HOME"] = path
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def apply_smoke_config(exp_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Fast end-to-end test settings (~minutes, not hours)."""
    exp_cfg["training"]["steps"] = 300
    exp_cfg["training"]["batch_size"] = 512
    exp_cfg["training"]["buffer_size"] = 50_000
    exp_cfg["training"]["buffer_refill_threshold"] = 0.0
    exp_cfg["training"]["log_every"] = 50
    exp_cfg["data"]["eval_images"] = 50
    if args.max_train_images is None:
        args.max_train_images = 200


def result_dir(
    results_root: Path,
    backbone_name: str,
    layer_depth: str,
    sae_arch: str,
    results_subdir: str | None = None,
) -> Path:
    base = results_root / results_subdir if results_subdir else results_root
    return base / backbone_name / layer_depth / sae_arch


def print_experiment_grid(combos: List[Dict[str, Any]]) -> None:
    print(f"Experiment grid ({len(combos)} runs):\n")
    print(f"{'#':>3}  {'backbone':<20} {'layer':<8} {'idx':>3}  {'sae_arch':<12}")
    print("-" * 55)
    for i, c in enumerate(combos, 1):
        print(
            f"{i:>3}  {c['backbone']:<20} {c['layer_depth']:<8} "
            f"{c['layer_index']:>3}  {c['sae_arch']:<12}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-backbone SAE experiments")
    parser.add_argument("--dry_run", action="store_true", help="Print grid and exit")
    parser.add_argument("--backbone", action="append", help="Filter to backbone(s)")
    parser.add_argument("--layer", action="append", dest="layer_depth", help="Filter layer depth")
    parser.add_argument("--sae_arch", action="append", help="Filter SAE architecture")
    parser.add_argument(
        "--post2025-cache-dir",
        default=None,
        help="Post-May 2025 patch cache (default: D:\\hirise_post2025_cache\\patches)",
    )
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--config", default=str(CONFIGS_DIR / "experiment.yaml"))
    parser.add_argument("--backbones_config", default=str(CONFIGS_DIR / "backbones.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument(
        "--reeval-only",
        action="store_true",
        help="Skip SAE training; re-run eval metrics on existing sae_weights.pt",
    )
    parser.add_argument(
        "--allow-prithvi-rgb-proxy",
        action="store_true",
        help=(
            "Allow Prithvi on RGB HiRISE by duplicating channels to 6 bands "
            "(not true multispectral; off by default)"
        ),
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail if any labeled image is corrupt (default: skip corrupt files)",
    )
    parser.add_argument(
        "--rescan-images",
        action="store_true",
        help="Re-scan all HiRISE files (ignore readable cache)",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Limit train images for debugging (eval unchanged unless --smoke)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast test: 300 SAE steps, 200 train / 50 eval images, small buffer",
    )
    parser.add_argument(
        "--results-subdir",
        default=None,
        help="Write under results/<subdir>/ (e.g. ablations/momo/d2_perdim)",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["scalar", "per_dim", "pca_proj", "zca_whiten"],
        default=None,
        help="SAE input normalization (default: training.preprocess_mode in config)",
    )
    parser.add_argument(
        "--allow-experimental-preprocess",
        action="store_true",
        help="Allow pca_proj / zca_whiten on non-MOMO backbones (MOMO-only by default)",
    )
    parser.add_argument(
        "--center-activations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Subtract per-dimension mean before SAE (per_dim mode only)",
    )
    parser.add_argument(
        "--dictionary-multiplier",
        type=int,
        default=None,
        help="Override TopK dictionary_multiplier (Matryoshka uses [1, m])",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=None,
        help="Weight on auxiliary dead-latent loss (default: 1.0)",
    )
    parser.add_argument(
        "--dataset",
        choices=("post2025_hirise", "hirise_post2025"),
        default=os.environ.get("SAE_DATASET", "post2025_hirise"),
        help="SAE training corpus (post-May 2025 cached patches only)",
    )
    return parser.parse_args()


def append_run_log(results_root: Path, message: str) -> None:
    log_path = results_root / "run.log"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def maybe_limit_train_set(train_set, max_images: Optional[int]):
    if max_images is None or max_images <= 0 or max_images >= len(train_set):
        return train_set
    return torch.utils.data.Subset(train_set, list(range(max_images)))


def run_single_experiment(
    combo: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    backbone_cfgs: Dict[str, Any],
    args: argparse.Namespace,
    split_info: Dict[str, Any],
    post2025_eval_entries: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    backbone_name = combo["backbone"]
    layer_depth = combo["layer_depth"]
    layer_index = combo["layer_index"]
    sae_arch = combo["sae_arch"]
    hidden_dim = combo["hidden_dim"]

    out_dir = result_dir(
        Path(args.results_dir),
        backbone_name,
        layer_depth,
        sae_arch,
        getattr(args, "results_subdir", None),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "sae_weights.pt"

    device = args.device or exp_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        tqdm.write("CUDA unavailable; falling back to CPU.")

    run_label = f"{backbone_name}/{layer_depth}/{sae_arch}"
    data_cfg = exp_cfg["data"]
    seed = int(exp_cfg.get("seed", 42))
    batch_size = int(data_cfg["activation_batch_size"])
    num_workers = int(data_cfg.get("dataloader_workers", 0))

    backbone = create_backbone(
        backbone_name,
        device=device,
        configs=backbone_cfgs,
        allow_prithvi_rgb_proxy=args.allow_prithvi_rgb_proxy,
    )
    transform = backbone.get_transform()
    cache_dir = Path(
        args.post2025_cache_dir
        or getattr(args, "hirise_post2025_cache_dir", None)
        or default_patch_cache_dir()
    )
    train_set, eval_set, split_info, _train_entries, eval_entries = build_post_may2025_datasets(
        cache_dir=cache_dir,
        transform=transform,
        train_fraction=float(data_cfg["train_fraction"]),
        seed=seed,
    )
    split_info = {**split_info, "eval_dataset": "marsbench"}
    post2025_eval_entries = eval_entries

    loaded_existing_checkpoint = False
    if args.reeval_only:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"--reeval-only: missing {weights_path} (train first or drop --reeval-only)"
            )
        tqdm.write(f"[REEVAL] {weights_path}")
        sae, payload = load_sae_checkpoint(weights_path, device)
        loaded_existing_checkpoint = True
    elif weights_path.exists() and not args.force:
        tqdm.write(f"[SKIP training] {weights_path} exists")
        sae, payload = load_sae_checkpoint(weights_path, device)
        loaded_existing_checkpoint = True
    else:
        train_set = maybe_limit_train_set(
            train_set, getattr(args, "max_train_images", None)
        )
        train_loader = activation_dataloader(
            train_set,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        sae_train_batch = int(exp_cfg["training"]["batch_size"])
        buffer = ActivationBuffer(
            capacity=int(exp_cfg["training"]["buffer_size"]),
            hidden_dim=hidden_dim,
            refill_threshold=float(exp_cfg["training"]["buffer_refill_threshold"]),
            sae_batch_size=sae_train_batch,
        )
        train_norm_scalar = stream_patch_activations_to_buffer(
            backbone=backbone,
            dataloader=train_loader,
            layer_index=layer_index,
            buffer=buffer,
            device=device,
            desc=f"[{run_label}] Stream train activations",
        )
        buffer.mark_refilled(training_step=0)
        train_cfg = exp_cfg["training"]
        preprocess_mode = (
            args.preprocess_mode
            if args.preprocess_mode is not None
            else train_cfg.get("preprocess_mode", "scalar")
        )
        if (
            preprocess_mode in EXPERIMENTAL_PREPROCESS_MODES
            and backbone_name != "momo"
            and not args.allow_experimental_preprocess
        ):
            raise ValueError(
                f"{preprocess_mode!r} is MOMO-only for now; pass "
                "--allow-experimental-preprocess to override."
            )
        center = (
            args.center_activations
            if args.center_activations is not None
            else train_cfg.get("center_activations", True)
        )
        norm_spec = fit_activation_norm(
            buffer, mode=preprocess_mode, center=bool(center)
        )
        if preprocess_mode == "scalar":
            norm_spec["norm_scalar"] = train_norm_scalar
        effective_dim = sae_input_dim(hidden_dim, norm_spec)
        tqdm.write(
            f"  buffer vectors={buffer.storage.shape[0]} "
            f"hidden_dim={hidden_dim} sae_d_in={effective_dim} "
            f"preprocess={preprocess_mode} "
            f"norm_scalar={norm_spec.get('norm_scalar', 1.0):.4f}"
            + (
                f" pca_k_95={norm_spec['pca_k_95']}"
                if preprocess_mode == "pca_proj"
                else ""
            )
        )
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        sae_cfg = dict(exp_cfg["sae_architectures"][sae_arch])
        if args.dictionary_multiplier is not None:
            mult = int(args.dictionary_multiplier)
            if sae_arch == "topk":
                sae_cfg["dictionary_multiplier"] = mult
            else:
                sae_cfg["nested_multipliers"] = [1, mult]
        training_cfg = dict(train_cfg)
        if args.aux_loss_weight is not None:
            training_cfg["aux_loss_weight"] = float(args.aux_loss_weight)

        sae = build_sae(sae_arch, effective_dim, sae_cfg, device)
        curve, train_stats = train_sae(
            sae=sae,
            buffer=buffer,
            backbone=backbone,
            train_loader=train_loader,
            layer_index=layer_index,
            cfg=training_cfg,
            device=device,
            norm_spec=norm_spec,
        )

        np.save(out_dir / "training_loss_curve.npy", curve)
        with (out_dir / "activation_stats.json").open("w", encoding="utf-8") as f:
            json.dump(train_stats, f, indent=2)

        meta = {
            "backbone": backbone_name,
            "layer_depth": layer_depth,
            "layer_index": layer_index,
            "sae_arch": sae_arch,
            "preprocess_mode": preprocess_mode,
            "hidden_dim": hidden_dim,
            "sae_d_in": effective_dim,
            "dictionary_multiplier": sae_cfg.get("dictionary_multiplier"),
            "nested_multipliers": sae_cfg.get("nested_multipliers"),
            "aux_loss_weight": training_cfg.get("aux_loss_weight", 1.0),
            "data_split": {
                "num_train": split_info["num_train"],
                "num_eval": split_info["num_eval"],
            },
        }
        if preprocess_mode == "pca_proj":
            meta["pca_k_95"] = int(norm_spec["pca_k_95"])
        if preprocess_mode == "zca_whiten":
            meta["zca_eps"] = float(norm_spec.get("zca_eps", 1e-3))
        save_sae_checkpoint(weights_path, sae, norm_spec, meta)
        tqdm.write(f"Saved SAE weights to {weights_path}")
        sae, payload = load_sae_checkpoint(weights_path, device)

    eval_norm_spec = norm_spec_for_eval(payload)

    n_eval = min(int(data_cfg["eval_images"]), len(eval_set))
    eval_indices = list(range(n_eval))
    eval_subset = torch.utils.data.Subset(eval_set, eval_indices)
    eval_entry_slice = (
        post2025_eval_entries[:n_eval]
        if post2025_eval_entries is not None
        else eval_set.entries[:n_eval]
    )

    eval_loader_plain = activation_dataloader(
        eval_subset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    indexed_eval = PostMay2025IndexedDataset(eval_entry_slice, transform=transform)
    eval_loader_indexed = activation_dataloader(
        indexed_eval,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    eval_acts = extract_patch_activations(
        backbone=backbone,
        dataloader=eval_loader_plain,
        layer_index=layer_index,
        device=device,
        desc=f"[{run_label}] Extract eval activations",
    )

    if loaded_existing_checkpoint:
        verify_checkpoint_norm_consistency(
            weights_path,
            backbone_name,
            layer_index,
            eval_acts,
            device,
        )

    eval_metrics = evaluate_sae_on_activations(
        sae=sae,
        activations=eval_acts,
        norm_spec=eval_norm_spec,
        device=device,
    )

    feature_matrix = compute_feature_activation_matrix(
        sae=sae,
        backbone=backbone,
        eval_dataset=eval_subset,
        eval_loader=eval_loader_indexed,
        layer_index=layer_index,
        norm_spec=eval_norm_spec,
        device=device,
    )
    np.save(out_dir / "feature_activation_matrix.npy", feature_matrix)

    eval_metrics["absorption_proxy"] = absorption_proxy(feature_matrix)
    with (out_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    row = {
        "backbone": backbone_name,
        "layer_depth": layer_depth,
        "layer_index": layer_index,
        "sae_arch": sae_arch,
        "preprocess_mode": payload.get("meta", {}).get(
            "preprocess_mode", payload.get("preprocess_mode", "scalar")
        ),
        "pca_k_95": payload.get("meta", {}).get("pca_k_95", float("nan")),
        "loss_recovered": eval_metrics["loss_recovered"],
        "loss_recovered_valid": eval_metrics.get("loss_recovered_valid", True),
        "mse_reduction_vs_zero": eval_metrics.get("mse_reduction_vs_zero", float("nan")),
        "mean_L0": eval_metrics["mean_L0"],
        "dead_fraction": eval_metrics["dead_fraction"],
        "absorption_proxy": eval_metrics["absorption_proxy"],
    }
    lr_note = "" if row["loss_recovered_valid"] else " (dense baseline invalid; see mse_reduction_vs_zero)"
    tqdm.write(
        f"  eval: loss_recovered={row['loss_recovered']:.4f}{lr_note} "
        f"mse_vs_zero={row['mse_reduction_vs_zero']:.3f} "
        f"L0={row['mean_L0']:.1f} dead={row['dead_fraction']:.3f} "
        f"absorption={row['absorption_proxy']:.3f}"
    )

    del backbone, sae
    torch.cuda.empty_cache()
    return row


def main() -> None:
    print("SAE-Experiments: parsing arguments...", flush=True)
    check_runtime_environment()
    args = parse_args()
    exp_cfg = load_yaml(Path(args.config))
    backbone_cfgs = load_backbone_configs(Path(args.backbones_config))

    if args.dry_run:
        combos = list_experiment_combinations(
            backbone_names=args.backbone,
            layer_depths=args.layer_depth,
            sae_archs=args.sae_arch,
            configs=backbone_cfgs,
        )
        print_experiment_grid(combos)
        return

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        apply_smoke_config(exp_cfg, args)
        print(
            "SMOKE MODE: 300 SAE steps, buffer=50k, "
            f"train<={args.max_train_images} images, eval=50 images"
        )

    print("Setting up HuggingFace cache...", flush=True)
    hf_cache = setup_hf_cache()
    print(f"HF cache: {hf_cache}", flush=True)

    try:
        _main_run(args, exp_cfg, backbone_cfgs, results_root)
    except Exception as exc:
        err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        append_run_log(results_root, f"FATAL\n{err_text}")
        raise


def _main_run(
    args: argparse.Namespace,
    exp_cfg: Dict[str, Any],
    backbone_cfgs: Dict[str, Any],
    results_root: Path,
) -> None:

    combos = list_experiment_combinations(
        backbone_names=args.backbone,
        layer_depths=args.layer_depth,
        sae_archs=args.sae_arch,
        configs=backbone_cfgs,
    )

    print(f"Planned runs: {len(combos)}", flush=True)
    for c in combos:
        print(f"  - {c['backbone']} / {c['layer_depth']} / {c['sae_arch']}", flush=True)

    cache_dir = Path(
        args.post2025_cache_dir
        or getattr(args, "hirise_post2025_cache_dir", None)
        or default_patch_cache_dir()
    )
    data_summary = validate_post_may2025_cache(cache_dir)
    print("SAE training corpus: post-May 2025 HiRISE patches (cache only):")
    print(f"  cache_dir:      {data_summary['cache_dir']}")
    print(f"  observations:   {data_summary['n_observations']}")
    print(f"  patches:        {data_summary['n_patches']}")
    print(f"  {data_summary['note']}")
    print("Evaluation corpus: Mars-Bench (labeled metrics via compute_saebench.py)")

    _, _, split_info, _train_entries, eval_entries = build_post_may2025_datasets(
        cache_dir=cache_dir,
        train_fraction=float(exp_cfg["data"]["train_fraction"]),
        seed=int(exp_cfg.get("seed", 42)),
    )
    args._post2025_eval_entries = eval_entries
    manifest_path = results_root / "post2025_patch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **split_info,
                "n_patch_entries": split_info["n_patches"],
            },
            f,
            indent=2,
        )
    with (results_root / "data_split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(
        f"  split manifest: {results_root / 'data_split_manifest.json'} "
        f"(train={split_info.get('num_train')} eval={split_info.get('num_eval')})"
    )

    if any(c["backbone"] == "prithvi_eo_2" for c in combos) and not args.allow_prithvi_rgb_proxy:
        print(
            "\nNote: prithvi_eo_2 is in the run list but HiRISE is RGB-only. "
            "Omit it, or pass --allow-prithvi-rgb-proxy (duplicates RGB to 6 bands).\n"
        )

    summary_rows: List[Dict[str, Any]] = []

    for combo in tqdm(combos, desc="Overall progress", unit="run"):
        tqdm.write(
            f"\n=== {combo['backbone']} / {combo['layer_depth']} "
            f"(layer {combo['layer_index']}) / {combo['sae_arch']} ==="
        )
        try:
            row = run_single_experiment(
                combo,
                exp_cfg,
                backbone_cfgs,
                args,
                split_info,
                post2025_eval_entries=getattr(args, "_post2025_eval_entries", None),
            )
            summary_rows.append(row)
        except Exception as exc:
            err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            tqdm.write(f"[ERROR] {combo}: {exc}")
            append_run_log(results_root, f"FAILED {combo}\n{err_text}")
            summary_rows.append(
                {
                    "backbone": combo["backbone"],
                    "layer_depth": combo["layer_depth"],
                    "layer_index": combo["layer_index"],
                    "sae_arch": combo["sae_arch"],
                    "loss_recovered": float("nan"),
                    "loss_recovered_valid": False,
                    "mse_reduction_vs_zero": float("nan"),
                    "mean_L0": float("nan"),
                    "dead_fraction": float("nan"),
                    "absorption_proxy": float("nan"),
                    "error": str(exc),
                }
            )

    summary_path = results_root / "summary_table.csv"
    new_df = pd.DataFrame(summary_rows)
    if summary_path.is_file() and len(combos) < 36:
        key = ["backbone", "layer_depth", "sae_arch"]
        old_df = pd.read_csv(summary_path)
        kept = old_df[~old_df[key].apply(tuple, axis=1).isin(new_df[key].apply(tuple, axis=1))]
        new_df = pd.concat([new_df, kept], ignore_index=True)
    new_df = new_df.sort_values(["backbone", "layer_depth", "sae_arch"]).reset_index(drop=True)
    new_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path} ({len(new_df)} rows)")


if __name__ == "__main__":
    main()
