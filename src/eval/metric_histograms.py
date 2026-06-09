"""Generate metric histogram PNGs from fair-protocol CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

META_COLS = {
    "backbone",
    "layer_depth",
    "sae_arch",
    "ablation_tag",
    "protocol",
    "preprocess_mode",
    "saebench_status",
    "results_subdir",
    "sparse_probe_eval_splits",
    "sparse_probe_split",
    "uniqueness_matrix_path",
    "sae_train_dataset",
    "eval_dataset",
}


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if col.endswith("_ok") or col.endswith("_beats_random"):
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_")


def plot_metric_across_backbones(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    bins: int = 20,
    alpha: float = 0.5,
    dpi: int = 150,
) -> bool:
    values_by_bb: dict[str, np.ndarray] = {}
    for backbone, group in df.groupby("backbone"):
        vals = pd.to_numeric(group[metric], errors="coerce").dropna().values
        if vals.size == 0:
            continue
        values_by_bb[str(backbone)] = vals
    if not values_by_bb:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for backbone, vals in sorted(values_by_bb.items()):
        ax.hist(vals, bins=bins, alpha=alpha, label=backbone, density=True)
    ax.set_title(f"{metric} (density, all runs)")
    ax.set_xlabel(metric)
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_backbone_metrics(
    row: pd.Series,
    metric_cols: list[str],
    out_path: Path,
    bins: int = 20,
    alpha: float = 0.5,
    dpi: int = 150,
) -> bool:
    vals: list[float] = []
    labels: list[str] = []
    for col in metric_cols:
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            vals.append(float(val))
            labels.append(col)
    if not vals:
        return False

    fig, ax = plt.subplots(figsize=(max(10, len(vals) * 0.35), 5))
    ax.bar(range(len(vals)), vals, alpha=min(1.0, alpha + 0.3))
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_title(f"{row['backbone']} — metric values (paper winner row)")
    ax.set_ylabel("value")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def generate_metric_histograms(
    csv_path: Path,
    output_dir: Path,
    paper_csv: Path | None = None,
    bins: int = 20,
    alpha: float = 0.5,
    dpi: int = 150,
) -> dict[str, int]:
    """
    Standalone histogram generation from fair CSV(s).

    Returns counts of per-metric and per-backbone plots written.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing input CSV {csv_path}")

    df = pd.read_csv(csv_path)
    if "backbone" not in df.columns:
        raise ValueError(f"{csv_path} missing 'backbone' column")

    metric_cols = numeric_metric_columns(df)
    per_metric_dir = output_dir / "per_metric"
    per_backbone_dir = output_dir / "per_backbone"

    n_metric = 0
    for metric in metric_cols:
        out_path = per_metric_dir / f"{safe_filename(metric)}_all_backbones.png"
        if plot_metric_across_backbones(df, metric, out_path, bins=bins, alpha=alpha, dpi=dpi):
            n_metric += 1

    n_backbone = 0
    if paper_csv and paper_csv.is_file():
        paper = pd.read_csv(paper_csv)
        for _, row in paper.iterrows():
            bb = str(row["backbone"])
            out_path = per_backbone_dir / f"{safe_filename(bb)}_all_metrics.png"
            if plot_backbone_metrics(row, metric_cols, out_path, bins=bins, alpha=alpha, dpi=dpi):
                n_backbone += 1

    return {"per_metric": n_metric, "per_backbone": n_backbone, "n_metrics": len(metric_cols)}
