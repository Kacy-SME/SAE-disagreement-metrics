#!/usr/bin/env python3
"""
Targeted MOMO SAE ablation: dict size × per-dim norm × aux loss weight.

Writes under results/ablations/momo/<tag>/momo/<layer>/<arch>/
Then runs SAEBench probing (no activation cache) into ablations/momo/saebench_scores.csv

Example (full grid, ~12 trains × ~30 min):
  python scripts/sweep_momo_sae.py --train --probe

Quick smoke (1 run, 300 steps via --smoke on run_experiments):
  python scripts/sweep_momo_sae.py --train --probe --smoke --limit 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe")
HIRISE_IMAGES = Path(r"D:\hirise_v3_2\images")
HIRISE_LABELS = Path(
    r"G:\My Drive\metrics&models\mars_data\hirise_v3_2\labels-map-proj-v3_2.txt"
)

DEFAULT_GRID = {
    "layer_depth": ["middle", "late"],
    "sae_arch": ["topk"],
    "dictionary_multiplier": [1, 2, 4],
    "preprocess_mode": ["per_dim"],
    "aux_loss_weight": [1.0, 3.0],
}


def ablation_tag(
    layer: str,
    arch: str,
    mult: int,
    preprocess: str,
    aux_w: float,
) -> str:
    pp_map = {
        "per_dim": "perdim",
        "scalar": "scalar",
        "pca_proj": "pcaproj",
        "zca_whiten": "zcawhiten",
    }
    pp = pp_map.get(preprocess, preprocess.replace("_", ""))
    return f"{layer}_{arch}_d{mult}_{pp}_aux{str(aux_w).replace('.', 'p')}"


def results_subdir(tag: str) -> str:
    return f"ablations/momo/{tag}"


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===", flush=True)
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def iter_grid(args: argparse.Namespace):
    layers = args.layer or DEFAULT_GRID["layer_depth"]
    archs = args.sae_arch or DEFAULT_GRID["sae_arch"]
    mults = args.dict_multiplier or DEFAULT_GRID["dictionary_multiplier"]
    modes = args.preprocess_mode or DEFAULT_GRID["preprocess_mode"]
    auxs = args.aux_loss_weight or DEFAULT_GRID["aux_loss_weight"]
    for layer, arch, mult, mode, aux in product(layers, archs, mults, modes, auxs):
        yield layer, arch, mult, mode, aux


def train_ablations(args: argparse.Namespace) -> list[dict]:
    py = str(args.python)
    records: list[dict] = []
    for i, (layer, arch, mult, mode, aux) in enumerate(iter_grid(args)):
        if args.limit is not None and i >= args.limit:
            break
        tag = ablation_tag(layer, arch, mult, mode, aux)
        sub = results_subdir(tag)
        cmd = [
            py,
            str(PROJECT_ROOT / "run_experiments.py"),
            "--force",
            "--backbone",
            "momo",
            "--layer",
            layer,
            "--sae_arch",
            arch,
            "--results-subdir",
            sub,
            "--preprocess-mode",
            mode,
            "--dictionary-multiplier",
            str(mult),
            "--aux-loss-weight",
            str(aux),
            "--hirise_images_dir",
            str(args.hirise_images),
            "--hirise_labels_file",
            str(args.hirise_labels),
        ]
        if args.smoke:
            cmd.append("--smoke")
        run_cmd(cmd, f"Train {tag}")
        records.append(
            {
                "tag": tag,
                "results_subdir": sub,
                "layer_depth": layer,
                "sae_arch": arch,
                "dictionary_multiplier": mult,
                "preprocess_mode": mode,
                "aux_loss_weight": aux,
            }
        )
    manifest = PROJECT_ROOT / "results" / "ablations" / "momo" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote manifest: {manifest} ({len(records)} runs)", flush=True)
    return records


def probe_ablations(args: argparse.Namespace, records: list[dict] | None = None) -> None:
    py = str(args.python)
    manifest_path = PROJECT_ROOT / "results" / "ablations" / "momo" / "manifest.json"
    if records is None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No manifest at {manifest_path}; run with --train first")
        with manifest_path.open(encoding="utf-8") as f:
            records = json.load(f)

    rows: list[dict] = []
    for rec in records:
        sub = rec["results_subdir"]
        results_root = PROJECT_ROOT / "results" / sub
        cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "compute_saebench.py"),
            "--extract",
            "--probe-landforms-only",
            "--probe-supplementary",
            "--no-cache-activations",
            "--results-dir",
            str(results_root),
            "--data-dir",
            str(PROJECT_ROOT / "results"),
            "--backbone",
            "momo",
            "--layer-depth",
            rec["layer_depth"],
            "--sae-arch",
            rec["sae_arch"],
        ]
        run_cmd(cmd, f"SAEBench {rec['tag']}")
        csv_path = results_root / "saebench_scores.csv"
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            eval_json = (
                results_root
                / "momo"
                / rec["layer_depth"]
                / rec["sae_arch"]
                / "eval_metrics.json"
            )
            if eval_json.is_file():
                with eval_json.open(encoding="utf-8") as f:
                    em = json.load(f)
                df["dead_fraction"] = em.get("dead_fraction")
                df["loss_recovered"] = em.get("loss_recovered")
            df["ablation_tag"] = rec["tag"]
            df["dictionary_multiplier"] = rec["dictionary_multiplier"]
            df["preprocess_mode"] = rec["preprocess_mode"]
            df["aux_loss_weight"] = rec["aux_loss_weight"]
            rows.append(df)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out_path = PROJECT_ROOT / "results" / "ablations" / "momo" / "saebench_scores.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nCombined scores: {out_path}", flush=True)
        sort_col = (
            "sparse_probe_f1_sae_latents"
            if "sparse_probe_f1_sae_latents" in out.columns
            else out.columns[0]
        )
        show_cols = [
            c
            for c in [
                "ablation_tag",
                "layer_depth",
                "sae_arch",
                "dictionary_multiplier",
                "aux_loss_weight",
                "saebench_status",
                "dead_fraction",
                "core_explained_variance",
                "sparse_probe_f1_sae_latents",
                "sparse_probe_f1_backbone",
                "sparse_probe_supp_f1_sae_latents",
            ]
            if c in out.columns
        ]
        if show_cols:
            print(
                out[show_cols].sort_values(sort_col, ascending=False).to_string(
                    index=False
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="MOMO SAE hyperparameter ablation sweep")
    parser.add_argument("--train", action="store_true", help="Run SAE training grid")
    parser.add_argument("--probe", action="store_true", help="Run SAEBench on ablation runs")
    parser.add_argument("--smoke", action="store_true", help="Pass --smoke to run_experiments")
    parser.add_argument("--limit", type=int, default=None, help="Max training runs")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--hirise-images", type=Path, default=HIRISE_IMAGES)
    parser.add_argument("--hirise-labels", type=Path, default=HIRISE_LABELS)
    parser.add_argument("--layer", action="append", default=None)
    parser.add_argument("--sae-arch", action="append", dest="sae_arch", default=None)
    parser.add_argument("--dict-multiplier", action="append", type=int, default=None)
    parser.add_argument("--preprocess-mode", action="append", default=None)
    parser.add_argument("--aux-loss-weight", action="append", type=float, default=None)
    args = parser.parse_args()

    if not args.train and not args.probe:
        parser.error("Pass --train and/or --probe")

    records = None
    if args.train:
        records = train_ablations(args)
    if args.probe:
        probe_ablations(args, records)


if __name__ == "__main__":
    main()
