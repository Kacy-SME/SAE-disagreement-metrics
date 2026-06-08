#!/usr/bin/env python3
"""
Fair SAE ablation across all backbones: same grid / same winner protocol.

Outputs: results/ablations/<backbone>/<tag>/<backbone>/<layer>/<arch>/
Combined CSV: results/ablations/saebench_fair.csv

Protocols
---------
grid (default): middle+late × topk+matryoshka × dict{1,2,4} × per_dim × aux{1,3}
  → pick best per (backbone, layer, arch) by test F1 with dead_fraction < max_dead

winner: fixed MOMO-tuned settings for quick parity
  → topk: per_dim, dict×1, aux×1
  → matryoshka: per_dim, dict×2 ([768,1536] nested; dict×1 is degenerate [768,768]), aux×1

Examples
--------
# Full fair grid on all backbones (~120 trains, days of GPU)
python scripts/sweep_sae_fair.py --train --probe

# Winner protocol only (20 trains: 5 backbones × 2 layers × 2 archs)
python scripts/sweep_sae_fair.py --protocol winner --train --probe

# MOMO matryoshka with same per_dim settings as winning topk run
python scripts/sweep_sae_fair.py --protocol winner --train --probe \\
  --backbone momo --layer middle --sae-arch matryoshka

# Smoke one run
python scripts/sweep_sae_fair.py --protocol winner --train --smoke --limit 1 \\
  --backbone momo --layer middle --sae-arch matryoshka
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

ALL_BACKBONES = [
    "mars_orbital_vit",
    "momo",
    "dinov3_sat493m",
    "satmae_pp",
    "croma",
]

GRID = {
    "layer_depth": ["middle", "late"],
    "sae_arch": ["topk", "matryoshka"],
    "dictionary_multiplier": [1, 2, 4],
    "preprocess_mode": ["per_dim"],
    "aux_loss_weight": [1.0, 3.0],
}

# topk d1; matryoshka d2 avoids nested [1,1] → [768,768]
WINNER = [
    ("topk", 1),
    ("matryoshka", 2),
]

# Prefer full retrain probe scores over duplicate winner-protocol rows.
SCORE_OVERRIDES: dict[tuple[str, str, str], str] = {
    ("momo", "middle", "topk"): "ablations/momo/middle_topk_d1_perdim_aux1p0_full",
    ("momo", "middle", "matryoshka"): "ablations/momo/middle_matryoshka_d2_perdim_aux1p0_full",
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


def results_subdir(backbone: str, tag: str) -> str:
    return f"ablations/{backbone}/{tag}"


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===", flush=True)
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def iter_runs(args: argparse.Namespace):
    backbones = args.backbone or ALL_BACKBONES
    if args.protocol == "winner":
        layers = args.layer or GRID["layer_depth"]
        auxs = [1.0]
        mode = "per_dim"
        for bb in backbones:
            for layer in layers:
                for arch, mult in WINNER:
                    if args.sae_arch and arch not in args.sae_arch:
                        continue
                    yield bb, layer, arch, mult, mode, auxs[0]
        return

    layers = args.layer or GRID["layer_depth"]
    archs = args.sae_arch or GRID["sae_arch"]
    mults = args.dict_multiplier or GRID["dictionary_multiplier"]
    modes = args.preprocess_mode or GRID["preprocess_mode"]
    auxs = args.aux_loss_weight or GRID["aux_loss_weight"]
    for bb in backbones:
        for layer, arch, mult, mode, aux in product(layers, archs, mults, modes, auxs):
            yield bb, layer, arch, mult, mode, aux


def train_ablations(args: argparse.Namespace) -> list[dict]:
    py = str(args.python)
    records: list[dict] = []
    for i, (bb, layer, arch, mult, mode, aux) in enumerate(iter_runs(args)):
        if args.limit is not None and i >= args.limit:
            break
        tag = ablation_tag(layer, arch, mult, mode, aux)
        sub = results_subdir(bb, tag)
        cmd = [
            py,
            str(PROJECT_ROOT / "run_experiments.py"),
            "--force",
            "--backbone",
            bb,
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
        run_cmd(cmd, f"Train {bb}/{tag}")
        records.append(
            {
                "backbone": bb,
                "tag": tag,
                "results_subdir": sub,
                "layer_depth": layer,
                "sae_arch": arch,
                "dictionary_multiplier": mult,
                "preprocess_mode": mode,
                "aux_loss_weight": aux,
                "protocol": args.protocol,
            }
        )

    manifest = PROJECT_ROOT / "results" / "ablations" / "fair_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if manifest.is_file() and not args.overwrite_manifest:
        with manifest.open(encoding="utf-8") as f:
            existing = json.load(f)
    merged = existing + records
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote manifest: {manifest} (+{len(records)} runs, total {len(merged)})", flush=True)
    return records


def probe_ablations(args: argparse.Namespace, records: list[dict] | None = None) -> None:
    py = str(args.python)
    manifest_path = PROJECT_ROOT / "results" / "ablations" / "fair_manifest.json"
    if records is None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No manifest at {manifest_path}; run with --train first")
        with manifest_path.open(encoding="utf-8") as f:
            records = json.load(f)
    if args.backbone:
        records = [r for r in records if r["backbone"] in args.backbone]

    rows: list[dict] = []
    for rec in records:
        bb = rec["backbone"]
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
            bb,
            "--layer-depth",
            rec["layer_depth"],
            "--sae-arch",
            rec["sae_arch"],
        ]
        run_cmd(cmd, f"SAEBench {bb}/{rec['tag']}")
        csv_path = results_root / "saebench_scores.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        eval_json = (
            results_root / bb / rec["layer_depth"] / rec["sae_arch"] / "eval_metrics.json"
        )
        if eval_json.is_file():
            with eval_json.open(encoding="utf-8") as f:
                em = json.load(f)
            df["dead_fraction"] = em.get("dead_fraction")
            df["loss_recovered"] = em.get("loss_recovered")
        for k in (
            "backbone",
            "ablation_tag",
            "protocol",
            "layer_depth",
            "sae_arch",
            "dictionary_multiplier",
            "preprocess_mode",
            "aux_loss_weight",
        ):
            if k == "ablation_tag":
                df[k] = rec["tag"]
            elif k in rec:
                df[k] = rec[k]
        rows.append(df)

    if not rows:
        print("No probe rows collected.", flush=True)
        return

    out = pd.concat(rows, ignore_index=True)
    out_path = PROJECT_ROOT / "results" / "ablations" / "saebench_fair.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file() and records:
        existing = pd.read_csv(out_path)
        probed_keys = {
            (r["backbone"], r["layer_depth"], r["sae_arch"]) for r in records
        }
        keep = ~existing.apply(
            lambda row: (row["backbone"], row["layer_depth"], row["sae_arch"])
            in probed_keys,
            axis=1,
        )
        out = pd.concat([existing[keep], out], ignore_index=True)
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
            "backbone",
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
            out[show_cols].sort_values(sort_col, ascending=False).to_string(index=False),
            flush=True,
        )

    rebuild_fair_tables(max_dead=args.max_dead)


def _fair_metric_columns(df: pd.DataFrame) -> list[str]:
    """Interpretability + extra metric columns to carry into paper tables."""
    prefixes = (
        "mono_",
        "tcav_",
        "causal_",
        "dec_ortho_",
        "mono_ms_",
        "geo_hier_",
        "spatial_cons_",
    )
    return [
        c
        for c in df.columns
        if c.startswith(prefixes) or c in ("ood_gap", "ood_gap_backbone")
    ]


def pick_best_per_group(
    df: pd.DataFrame,
    max_dead: float = 0.40,
    require_all_groups: bool = False,
) -> pd.DataFrame:
    f1 = "sparse_probe_f1_sae_latents"
    if f1 not in df.columns:
        return df.head(0)
    keys = ["backbone", "layer_depth", "sae_arch"]
    if not all(k in df.columns for k in keys):
        return df.head(0)

    work = df.copy()
    if "dead_fraction" in work.columns:
        work["dead_ok"] = work["dead_fraction"].isna() | (work["dead_fraction"] <= max_dead)
    else:
        work["dead_ok"] = True
    if "core_explained_variance" in work.columns:
        work["core_ev_ok"] = work["core_explained_variance"].isna() | (
            work["core_explained_variance"] > 0
        )
    else:
        work["core_ev_ok"] = True

    picked: list[pd.Series] = []
    for _, g in work.groupby(keys, dropna=False):
        preferred = g[g["dead_ok"] & g["core_ev_ok"]]
        if preferred.empty:
            preferred = g
        picked.append(preferred.loc[preferred[f1].idxmax()])

    out = pd.DataFrame(picked)
    base_cols = [
        "backbone",
        "layer_depth",
        "sae_arch",
        "ablation_tag",
        "dictionary_multiplier",
        "aux_loss_weight",
        "dead_fraction",
        "dead_ok",
        "core_explained_variance",
        "core_ev_ok",
        f1,
        "sparse_probe_f1_backbone",
        "sparse_probe_supp_f1_sae_latents",
    ]
    metric_cols = _fair_metric_columns(out)
    cols = [c for c in base_cols if c in out.columns] + [
        c for c in metric_cols if c not in base_cols
    ]
    out = out[cols].sort_values(["backbone", "layer_depth", "sae_arch"])

    if require_all_groups:
        expected = work[keys].drop_duplicates().shape[0]
        if len(out) < expected:
            missing = expected - len(out)
            print(
                f"WARNING: {missing} (backbone, layer, arch) groups missing from fair table",
                flush=True,
            )
    return out


def pick_best_per_backbone(
    df: pd.DataFrame,
    max_dead: float = 0.40,
) -> pd.DataFrame:
    f1 = "sparse_probe_f1_sae_latents"
    if f1 not in df.columns or "backbone" not in df.columns:
        return df.head(0)

    work = df.copy()
    if "dead_ok" not in work.columns and "dead_fraction" in work.columns:
        work["dead_ok"] = work["dead_fraction"].isna() | (work["dead_fraction"] <= max_dead)
    if "core_ev_ok" not in work.columns and "core_explained_variance" in work.columns:
        work["core_ev_ok"] = work["core_explained_variance"].isna() | (
            work["core_explained_variance"] > 0
        )

    picked: list[pd.Series] = []
    for _, g in work.groupby("backbone", dropna=False):
        preferred = g[g.get("dead_ok", True) & g.get("core_ev_ok", True)]
        if preferred.empty:
            preferred = g
        picked.append(preferred.loc[preferred[f1].idxmax()])
    out = pd.DataFrame(picked)
    return out.sort_values("backbone")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair SAE ablation across all backbones")
    parser.add_argument(
        "--protocol",
        choices=("grid", "winner"),
        default="grid",
        help="grid=full search; winner=fixed per_dim settings (MOMO-tuned topk + matryoshka d2)",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--rebuild-tables",
        action="store_true",
        help="Rebuild fair CSVs from saebench_fair.csv (+ MOMO _full overrides)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-manifest", action="store_true")
    parser.add_argument("--max-dead", type=float, default=0.40, help="Best-pick dead_fraction cap")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--hirise-images", type=Path, default=HIRISE_IMAGES)
    parser.add_argument("--hirise-labels", type=Path, default=HIRISE_LABELS)
    parser.add_argument("--backbone", action="append", default=None)
    parser.add_argument("--layer", action="append", default=None)
    parser.add_argument("--sae-arch", action="append", dest="sae_arch", default=None)
    parser.add_argument("--dict-multiplier", action="append", type=int, default=None)
    parser.add_argument("--preprocess-mode", action="append", default=None)
    parser.add_argument("--aux-loss-weight", action="append", type=float, default=None)
    args = parser.parse_args()

    if not args.train and not args.probe and not args.rebuild_tables:
        parser.error("Pass --train, --probe, and/or --rebuild-tables")

    records = None
    if args.train:
        records = train_ablations(args)
    if args.probe:
        probe_ablations(args, records)
    if args.rebuild_tables:
        rebuild_fair_tables(max_dead=args.max_dead)


def load_saebench_row(results_subdir: str) -> dict | None:
    csv_path = PROJECT_ROOT / "results" / results_subdir / "saebench_scores.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def apply_score_overrides(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for (bb, layer, arch), sub in SCORE_OVERRIDES.items():
        row = load_saebench_row(sub)
        if row is None:
            continue
        mask = (
            (out["backbone"] == bb)
            & (out["layer_depth"] == layer)
            & (out["sae_arch"] == arch)
        )
        if not mask.any():
            continue
        skip_cols = set(_fair_metric_columns(out)) | {
            "saebench_status",
            "dead_ok",
            "core_ev_ok",
            "ood_gap",
            "ood_gap_backbone",
        }
        for col, val in row.items():
            if col not in out.columns or col in skip_cols:
                continue
            if pd.isna(val):
                continue
            try:
                out.loc[mask, col] = val
            except (TypeError, ValueError):
                continue
        if "ablation_tag" in out.columns:
            tag = sub.split("/")[-1]
            out.loc[mask, "ablation_tag"] = tag
    return out


def rebuild_fair_tables(max_dead: float = 0.40) -> None:
    fair_path = PROJECT_ROOT / "results" / "ablations" / "saebench_fair.csv"
    if not fair_path.is_file():
        raise FileNotFoundError(f"Missing {fair_path}; run winner protocol probe first.")
    df = pd.read_csv(fair_path)
    df = apply_score_overrides(df)
    df.to_csv(fair_path, index=False)
    print(f"Updated {fair_path} ({len(df)} rows)", flush=True)

    all_groups = pick_best_per_group(df, max_dead=max_dead, require_all_groups=True)
    all_path = PROJECT_ROOT / "results" / "ablations" / "saebench_fair_all.csv"
    all_groups.to_csv(all_path, index=False)
    print(f"All groups ({len(all_groups)} rows): {all_path}", flush=True)

    paper = pick_best_per_backbone(all_groups, max_dead=max_dead)
    paper_path = PROJECT_ROOT / "results" / "ablations" / "saebench_fair_paper.csv"
    paper.to_csv(paper_path, index=False)
    print(f"Paper picks ({len(paper)} backbones): {paper_path}", flush=True)
    if not paper.empty:
        show = [
            c
            for c in [
                "backbone",
                "layer_depth",
                "sae_arch",
                "ablation_tag",
                "dead_fraction",
                "core_explained_variance",
                "sparse_probe_f1_sae_latents",
                "sparse_probe_f1_backbone",
                "sparse_probe_supp_f1_sae_latents",
                "ood_gap",
                "mono_mean_purity_top20_official",
                "tcav_mean_max_cos_per_class",
                "dec_ortho_mean",
                "mono_ms_mean",
                "geo_hier_mean",
                "spatial_cons_mean",
                "dead_ok",
                "core_ev_ok",
            ]
            if c in paper.columns
        ]
        print(paper[show].to_string(index=False), flush=True)

    # Back-compat name for the complete per-group table
    best_path = PROJECT_ROOT / "results" / "ablations" / "saebench_fair_best.csv"
    all_groups.to_csv(best_path, index=False)
    print(f"Also wrote {best_path}", flush=True)


if __name__ == "__main__":
    main()
