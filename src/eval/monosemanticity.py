"""
Monosemanticity metrics inspired by:
  Bricken et al., "Towards Monosemanticity" (2023)
  https://transformer-circuits.pub/2023/monosemantic-features/

Vision / HiRISE adaptations:
  - Feature density (paper appendix: fraction of examples with nonzero activation)
  - Label purity on top-firing images (proxy for "interpretable activation conditions")
  - Activation-interval label consistency (paper samples 11 intervals across the spectrum)

Not implemented here (require LM logits or API):
  - Human rubric (0–14) — manual on exported top-image panels
  - Automated interpretability — logit weight prediction (N/A for ViT patch features)
  - Automated interpretability — activation Spearman / detection score (SAEBench-style LLM judge)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def feature_density_per_latent(matrix: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Fraction of eval images where latent activation > eps."""
    n_images = matrix.shape[1]
    if n_images == 0:
        return np.array([])
    active = matrix > eps
    return active.sum(axis=1).astype(np.float64) / n_images


def label_purity_topk(
    matrix: np.ndarray,
    labels: List[str],
    top_k: int = 20,
) -> np.ndarray:
    """
    Per latent: among top-k images by max activation, fraction sharing the modal label.
    """
    n_features, n_images = matrix.shape
    k = min(top_k, n_images)
    purities = np.zeros(n_features, dtype=np.float64)
    for i in range(n_features):
        if k == 0:
            purities[i] = 0.0
            continue
        top_idx = np.argsort(-matrix[i])[:k]
        top_labels = [labels[j] for j in top_idx if matrix[i, j] > 0]
        if not top_labels:
            purities[i] = 0.0
            continue
        _, counts = np.unique(top_labels, return_counts=True)
        purities[i] = counts.max() / len(top_labels)
    return purities


def interval_label_consistency(
    matrix: np.ndarray,
    labels: List[str],
    n_intervals: int = 11,
) -> np.ndarray:
    """
    Paper-style: split nonzero activation range into intervals; score whether
    labels in each interval agree with the modal label of the top interval.
    Returns per-latent mean consistency in [0, 1].
    """
    n_features = matrix.shape[0]
    scores = np.zeros(n_features, dtype=np.float64)
    for i in range(n_features):
        row = matrix[i]
        nz = row[row > 0]
        if nz.size < 3:
            scores[i] = float("nan")
            continue
        a_min, a_max = float(nz.min()), float(nz.max())
        if a_max <= a_min:
            scores[i] = 1.0
            continue
        edges = np.linspace(a_min, a_max, n_intervals + 1)
        interval_labels: List[str] = []
        for b in range(n_intervals):
            lo, hi = edges[b], edges[b + 1]
            if b < n_intervals - 1:
                mask = (row >= lo) & (row < hi) & (row > 0)
            else:
                mask = (row >= lo) & (row <= hi) & (row > 0)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            # Representative image: highest activation in interval
            best = idxs[np.argmax(row[idxs])]
            interval_labels.append(labels[best])
        if len(interval_labels) < 2:
            scores[i] = float("nan")
            continue
        modal = max(set(interval_labels), key=interval_labels.count)
        agree = sum(1 for lab in interval_labels if lab == modal)
        scores[i] = agree / len(interval_labels)
    return scores


def summarize_latent_stats(
    densities: np.ndarray,
    purities: np.ndarray,
    interval_scores: np.ndarray,
    ultralow_threshold: float = 1e-6,
) -> Dict[str, float]:
    """Run-level aggregates aligned with paper global analysis themes."""
    alive = densities > 0
    n = len(densities)
    if n == 0:
        return {}

    valid_interval = np.isfinite(interval_scores)
    alive_interval = alive & valid_interval

    return {
        "n_latents": float(n),
        "fraction_alive": float(alive.mean()),
        "fraction_dead": float(1.0 - alive.mean()),
        "median_feature_density": float(np.median(densities[alive])) if alive.any() else 0.0,
        "min_feature_density_alive": float(densities[alive].min()) if alive.any() else 0.0,
        "fraction_ultralow_density": float((densities[alive] < ultralow_threshold).sum() / max(alive.sum(), 1)),
        "mean_label_purity_top20": float(np.nanmean(purities[alive])) if alive.any() else 0.0,
        "median_label_purity_top20": float(np.nanmedian(purities[alive])) if alive.any() else 0.0,
        "mean_interval_label_consistency": float(np.nanmean(interval_scores[alive_interval]))
        if alive_interval.any()
        else float("nan"),
        "fraction_high_purity_top20": float((purities[alive] >= 0.8).mean()) if alive.any() else 0.0,
    }


def load_eval_labels(
    split_manifest: Path,
    readable_cache: Path,
    n_eval_images: int,
) -> List[str]:
    with split_manifest.open(encoding="utf-8") as f:
        split = json.load(f)
    eval_paths = split["eval_rel_paths"][:n_eval_images]
    with readable_cache.open(encoding="utf-8") as f:
        cache = json.load(f)
    labels_by_path = cache.get("labels_by_path", {})
    return [labels_by_path.get(rel, "unknown") for rel in eval_paths]


def score_run_from_matrix(
    matrix: np.ndarray,
    labels: List[str],
    top_k: int = 20,
    n_intervals: int = 11,
) -> Dict[str, Any]:
    densities = feature_density_per_latent(matrix)
    purities = label_purity_topk(matrix, labels, top_k=top_k)
    interval_scores = interval_label_consistency(matrix, labels, n_intervals=n_intervals)
    summary = summarize_latent_stats(densities, purities, interval_scores)
    return {
        **summary,
        "feature_density": densities,
        "label_purity_top20": purities,
        "interval_label_consistency": interval_scores,
    }


def score_run_directory(
    run_dir: Path,
    split_manifest: Path,
    readable_cache: Path,
    n_eval_images: int = 500,
    matrix_name: str = "feature_activation_matrix.npy",
) -> Optional[Dict[str, Any]]:
    matrix_path = run_dir / matrix_name
    if not matrix_path.is_file():
        return None
    matrix = np.load(matrix_path)
    n_cols = min(matrix.shape[1], n_eval_images)
    matrix = matrix[:, :n_cols]
    labels = load_eval_labels(split_manifest, readable_cache, n_cols)
    if len(labels) != n_cols:
        raise ValueError(f"Label count {len(labels)} != matrix columns {n_cols}")
    return score_run_from_matrix(matrix, labels)
