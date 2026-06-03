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
from src.data.hirise import (  # noqa: E402
    build_hirise_splits,
    build_readable_index,
    build_split_info,
    save_split_manifest,
    validate_hirise_paths,
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
from src.sae.trainer import (  # noqa: E402
    build_sae,
    load_sae_checkpoint,
    save_sae_checkpoint,
    train_sae,
    verify_checkpoint_norm_consistency,
)
from src.sae.utils import sae_inference_norm_scalar  # noqa: E402

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
) -> Path:
    return results_root / backbone_name / layer_depth / sae_arch


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
        "--hirise_images_dir",
        default=os.environ.get("HIRISE_IMAGES_DIR", DEFAULT_PATHS["hirise_images_dir"]),
        help="Directory of real HiRISE v3.2 JPGs (train and eval both come from here)",
    )
    parser.add_argument(
        "--hirise_labels_file",
        default=os.environ.get("HIRISE_LABELS_FILE", DEFAULT_PATHS["hirise_labels_file"]),
        help="labels-map-proj-v3_2.txt — no separate test-labels path needed",
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
    readable_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    backbone_name = combo["backbone"]
    layer_depth = combo["layer_depth"]
    layer_index = combo["layer_index"]
    sae_arch = combo["sae_arch"]
    hidden_dim = combo["hidden_dim"]

    out_dir = result_dir(Path(args.results_dir), backbone_name, layer_depth, sae_arch)
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
    train_set, eval_set, _, _ = build_hirise_splits(
        images_dir=args.hirise_images_dir,
        labels_file=args.hirise_labels_file,
        transform=transform,
        train_fraction=float(data_cfg["train_fraction"]),
        seed=seed,
        readable_samples=readable_samples,
    )

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
        tqdm.write(
            f"  buffer vectors={buffer.storage.shape[0]} "
            f"hidden_dim={hidden_dim} norm_scalar={train_norm_scalar:.4f}"
        )
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        sae = build_sae(sae_arch, hidden_dim, exp_cfg["sae_architectures"][sae_arch], device)
        curve, train_stats = train_sae(
            sae=sae,
            buffer=buffer,
            backbone=backbone,
            train_loader=train_loader,
            layer_index=layer_index,
            cfg=exp_cfg["training"],
            device=device,
            norm_scalar=train_norm_scalar,
        )

        np.save(out_dir / "training_loss_curve.npy", curve)
        with (out_dir / "activation_stats.json").open("w", encoding="utf-8") as f:
            json.dump(train_stats, f, indent=2)

        meta = {
            "backbone": backbone_name,
            "layer_depth": layer_depth,
            "layer_index": layer_index,
            "sae_arch": sae_arch,
            "data_split": {
                "num_train": split_info["num_train"],
                "num_eval": split_info["num_eval"],
            },
        }
        save_sae_checkpoint(weights_path, sae, train_norm_scalar, meta)
        tqdm.write(f"Saved SAE weights to {weights_path}")
        sae, payload = load_sae_checkpoint(weights_path, device)

    # SAE weights have normalization folded in; eval uses raw backbone activations.
    eval_norm_scalar = sae_inference_norm_scalar(payload)

    n_eval = min(int(data_cfg["eval_images"]), len(eval_set))
    eval_subset = torch.utils.data.Subset(eval_set, list(range(n_eval)))

    eval_loader_plain = activation_dataloader(
        eval_subset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    eval_loader_indexed = activation_dataloader(
        IndexedSubset(eval_subset),
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
        norm_scalar=eval_norm_scalar,
        device=device,
    )

    feature_matrix = compute_feature_activation_matrix(
        sae=sae,
        backbone=backbone,
        eval_dataset=eval_subset,
        eval_loader=eval_loader_indexed,
        layer_index=layer_index,
        norm_scalar=eval_norm_scalar,
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

    cache_path = results_root / "hirise_readable_cache.json"

    print(f"Planned runs: {len(combos)}", flush=True)
    for c in combos:
        print(f"  - {c['backbone']} / {c['layer_depth']} / {c['sae_arch']}", flush=True)

    data_summary = validate_hirise_paths(
        args.hirise_images_dir,
        args.hirise_labels_file,
        cache_path=cache_path,
        strict=args.strict_data,
        rescan=args.rescan_images,
    )
    print("HiRISE data validated (real images only):")
    print(f"  images_dir:       {data_summary['images_dir']}")
    print(f"  labels_file:      {data_summary['labels_file']}")
    print(f"  files on disk:    {data_summary['num_images_listed']}")
    print(f"  readable images:  {data_summary['num_images_readable']}")
    if data_summary["num_skipped"]:
        print(f"  skipped (corrupt): {data_summary['num_skipped']}")
        if data_summary.get("skipped_sample"):
            print(f"  example skipped:   {data_summary['skipped_sample'][0]}")
    print(f"  {data_summary['note']}")

    cached_samples = None
    if cache_path.is_file() and not args.rescan_images and not args.strict_data:
        from src.data.hirise import load_readable_samples_from_cache

        cached_samples = load_readable_samples_from_cache(
            cache_path, args.hirise_images_dir, args.hirise_labels_file
        )
    if cached_samples is not None:
        readable_samples, skipped, _ = cached_samples
    else:
        readable_samples, skipped, _ = build_readable_index(
            args.hirise_images_dir,
            args.hirise_labels_file,
            cache_path=cache_path,
            strict=args.strict_data,
            rescan=args.rescan_images,
            show_progress=True,
        )
    readable_samples.sort(key=lambda s: s["rel_path"])
    if skipped:
        with (results_root / "skipped_hirise_images.json").open("w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2)
        print(f"  skipped list: {results_root / 'skipped_hirise_images.json'}")

    split_info = build_split_info(
        readable_samples,
        train_fraction=float(exp_cfg["data"]["train_fraction"]),
        seed=int(exp_cfg.get("seed", 42)),
        images_dir=args.hirise_images_dir,
        labels_file=args.hirise_labels_file,
    )
    save_split_manifest(results_root / "data_split_manifest.json", split_info)
    print(
        f"  split: {split_info['num_train']} train / {split_info['num_eval']} eval "
        f"(manifest: {results_root / 'data_split_manifest.json'})"
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
                combo, exp_cfg, backbone_cfgs, args, split_info, readable_samples
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
