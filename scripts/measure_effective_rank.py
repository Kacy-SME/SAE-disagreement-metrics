#!/usr/bin/env python3
"""
Measure effective rank of backbone activations across the fair protocol grid.

Uses official-probe train activations (mean-pooled per image) from
results/probe_activation_cache/{backbone}/{layer}/train_train_activations.pt
when available; pass --extract to build missing caches from ViT forwards.

Output: results/diagnostics/effective_rank.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones.registry import (  # noqa: E402
    create_backbone,
    list_experiment_combinations,
    load_backbone_configs,
)
from src.data.hirise import (  # noqa: E402
    HiRISEDataset,
    filter_readable_official_samples,
    load_official_split_table,
)
from src.eval.saebench_sparse_probing import _pool_image_activations  # noqa: E402
from src.extract.activations import activation_dataloader, extract_patch_activations  # noqa: E402
from src.paths import CONFIGS_DIR, RESULTS_DIR  # noqa: E402
from src.sae.effective_rank import summarize_effective_rank  # noqa: E402

ALL_BACKBONES = [
    "croma",
    "dinov3_sat493m",
    "mars_orbital_vit",
    "momo",
    "satmae_pp",
]
FAIR_LAYERS = ["middle", "late"]
DEFAULT_LABELS = Path(
    r"G:\My Drive\metrics&models\mars_data\hirise_v3_2\labels-map-proj-v3_2.txt"
)
DEFAULT_IMAGES = Path(r"D:\hirise_v3_2\images")
PYTHON = Path(r"c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "experiment.yaml")
    parser.add_argument("--backbones-config", type=Path, default=CONFIGS_DIR / "backbones.yaml")
    parser.add_argument("--backbone", action="append", default=None)
    parser.add_argument("--layer", action="append", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extract", action="store_true", help="Extract missing probe caches")
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS,
        help="HiRISE labels file for official train split",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES,
        help="HiRISE images root",
    )
    parser.add_argument(
        "--allow-prithvi-rgb-proxy",
        action="store_true",
        help="Unused for fair backbones; kept for CLI parity",
    )
    return parser.parse_args()


def layer_index_for(backbone: str, layer_depth: str, backbone_cfgs: dict) -> int:
    for combo in list_experiment_combinations(
        backbone_names=[backbone],
        layer_depths=[layer_depth],
        sae_archs=["topk"],
        configs=backbone_cfgs,
    ):
        return int(combo["layer_index"])
    raise KeyError(f"No layer index for {backbone}/{layer_depth}")


def probe_train_cache_path(results_dir: Path, backbone: str, layer_depth: str) -> Path:
    return (
        results_dir
        / "probe_activation_cache"
        / backbone
        / layer_depth
        / "train_train_activations.pt"
    )


def load_readable_set(results_dir: Path) -> set[str]:
    cache_path = results_dir / "hirise_readable_cache.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing readable cache: {cache_path}")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return set(payload.get("readable_rel_paths", []))


@torch.no_grad()
def extract_official_train_activations(
    backbone: str,
    layer_depth: str,
    images_dir: Path,
    labels_file: Path,
    readable_set: set[str],
    exp_cfg: dict,
    backbone_cfgs: dict,
    device: str,
) -> tuple[torch.Tensor, int]:
    layer_index = layer_index_for(backbone, layer_depth, backbone_cfgs)
    split_rows = load_official_split_table(str(labels_file))
    train_samples = filter_readable_official_samples(
        split_rows, readable_set, ["train"], landforms_only=True
    )
    if not train_samples:
        raise RuntimeError(f"No readable official train samples for {backbone}/{layer_depth}")

    backbone_model = create_backbone(
        backbone,
        device=device,
        configs=backbone_cfgs,
        allow_prithvi_rgb_proxy=False,
    )
    transform = backbone_model.get_transform()
    dataset = HiRISEDataset(
        images_dir=str(images_dir),
        labels_file=str(labels_file),
        transform=transform,
        samples=[
            {
                "path": images_dir / s["rel_path"],
                "rel_path": s["rel_path"],
                "label": s["label"],
            }
            for s in train_samples
        ],
    )
    loader = activation_dataloader(
        dataset,
        batch_size=int(exp_cfg["data"]["activation_batch_size"]),
        num_workers=int(exp_cfg["data"].get("dataloader_workers", 0)),
    )
    acts = extract_patch_activations(
        backbone=backbone_model,
        dataloader=loader,
        layer_index=layer_index,
        device=device,
        desc=f"Extract train {backbone}/{layer_depth}",
    ).cpu().float()
    del backbone_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return acts, len(train_samples)


def load_pooled_train_matrix(
    cache_path: Path,
) -> tuple[torch.Tensor, int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    acts = payload["activations"].float()
    n_images = int(payload.get("n_images", len(payload.get("labels", []))))
    if n_images <= 0:
        raise ValueError(f"Invalid n_images in {cache_path}")
    patches_per_image = acts.shape[0] // n_images
    pooled = _pool_image_activations(acts, n_images, patches_per_image)
    return pooled, n_images


def measure_one(
    backbone: str,
    layer_depth: str,
    args: argparse.Namespace,
    exp_cfg: dict,
    backbone_cfgs: dict,
    readable_set: set[str],
) -> dict:
    cache_path = probe_train_cache_path(args.results_dir, backbone, layer_depth)
    if cache_path.is_file():
        pooled, n_images = load_pooled_train_matrix(cache_path)
        source = "cache"
    elif args.extract:
        acts, n_images = extract_official_train_activations(
            backbone,
            layer_depth,
            args.images_dir,
            args.labels_file,
            readable_set,
            exp_cfg,
            backbone_cfgs,
            args.device,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "activations": acts,
                "labels": [""] * n_images,
                "n_images": n_images,
            },
            cache_path,
        )
        patches_per_image = acts.shape[0] // n_images
        pooled = _pool_image_activations(acts, n_images, patches_per_image)
        source = "extracted"
    else:
        raise FileNotFoundError(
            f"Missing {cache_path}; rerun with --extract to build probe caches"
        )

    stats = summarize_effective_rank(pooled)
    row = {
        "backbone": backbone,
        "layer_depth": layer_depth,
        "n_train_images": n_images,
        "source": source,
        **stats,
    }
    return row


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA unavailable; using CPU.", flush=True)

    exp_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    backbone_cfgs = load_backbone_configs(args.backbones_config)
    readable_set = load_readable_set(args.results_dir)

    backbones = args.backbone or ALL_BACKBONES
    layers = args.layer or FAIR_LAYERS
    rows: list[dict] = []

    for backbone in backbones:
        for layer_depth in layers:
            label = f"{backbone}/{layer_depth}"
            try:
                row = measure_one(
                    backbone,
                    layer_depth,
                    args,
                    exp_cfg,
                    backbone_cfgs,
                    readable_set,
                )
                rows.append(row)
                tqdm.write(
                    f"{label}: PR={row['participation_ratio']:.1f} "
                    f"k_95={row['k_95']} / d={row['hidden_dim']} "
                    f"({row['source']})"
                )
            except Exception as exc:
                tqdm.write(f"[ERROR] {label}: {exc}")

    if not rows:
        raise SystemExit("No effective-rank rows computed.")

    out_dir = args.results_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "effective_rank.csv"
    df = pd.DataFrame(rows).sort_values(["backbone", "layer_depth"])
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
