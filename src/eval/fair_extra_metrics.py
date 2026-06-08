"""
Extra SAE metrics from cached activations + decoder weights (no ViT re-extract).

- dec_ortho: decoder column cosine similarity
- mono_ms: monosemanticity via top-k image embedding similarity
- geo_hier: geological group structure in per-class latent profiles
- spatial: patch-level spatial consistency from cached patch tokens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy as scipy_entropy

from src.eval.saebench_interpretability import build_image_activation_matrix
from src.eval.saebench_sparse_probing import _pool_image_activations
from src.sae.activation_norm import preprocess_activations

# HiRISE v3.2 class IDs mapped to geological groups (user taxonomy; gully/mass_wasting N/A)
GEO_GROUPS: Dict[str, Set[str]] = {
    "aeolian": {"2", "4"},  # dark dune, bright dune
    "impact": {"1", "5"},  # crater, impact ejecta
    "slope": {"3"},  # slope streak (slope_streak)
}
GEO_GROUP_CLASSES: Set[str] = {c for g in GEO_GROUPS.values() for c in g}


def patches_per_image(backbone_cfg: dict) -> int:
    ps = int(backbone_cfg["patch_size"])
    isz = int(backbone_cfg["image_size"])
    return (isz // ps) ** 2


def patch_grid_shape(backbone_cfg: dict) -> Tuple[int, int]:
    side = int(backbone_cfg["image_size"]) // int(backbone_cfg["patch_size"])
    return side, side


def compute_decoder_orthogonality(sae: nn.Module) -> Dict[str, float]:
    """Mean / p95 pairwise cosine similarity of decoder columns (lower = more orthogonal)."""
    w = sae.decoder.weight.detach().cpu().float()  # (d_in, dict_size)
    cols = w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)
    sim = cols.T @ cols
    n = sim.shape[0]
    if n < 2:
        return {"dec_ortho_mean": float("nan"), "dec_ortho_p95": float("nan")}
    tri = torch.triu_indices(n, n, offset=1)
    pairs = sim[tri[0], tri[1]].numpy()
    return {
        "dec_ortho_mean": float(pairs.mean()),
        "dec_ortho_p95": float(np.percentile(pairs, 95)),
    }


def _mean_pooled_backbone_embeddings(
    backbone_acts: torch.Tensor,
    n_images: int,
    patches_per_image: int,
    norm_spec: dict,
) -> np.ndarray:
    acts_cpu = preprocess_activations(backbone_acts.float(), norm_spec).cpu()
    pooled = _pool_image_activations(acts_cpu, n_images, patches_per_image)
    return pooled.numpy()


def _pairwise_cosine_mean(vectors: np.ndarray) -> float:
    n = vectors.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-8, None)
    sim = normed @ normed.T
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def _class_mean_scalars(matrix: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Per-class mean activation for one latent (matrix row)."""
    out: Dict[str, float] = {}
    for cls in GEO_GROUP_CLASSES:
        idx = [i for i, lab in enumerate(labels) if str(lab).strip() == cls]
        if idx:
            out[cls] = float(matrix[idx].mean())
    return out


def _scalar_cosine(a: float, b: float) -> float:
    denom = abs(a) * abs(b)
    if denom < 1e-12:
        return 0.0
    return float(a * b / denom)


def compute_mono_ms(
    sae: nn.Module,
    test_acts: torch.Tensor,
    test_labels: List[str],
    norm_spec: dict,
    device: str,
    patches_per_image: int,
    top_k: int = 16,
    encode_chunk: int = 512,
) -> Dict[str, float]:
    """
    Mean pairwise cosine similarity among top-k image backbone embeddings per live latent.
    """
    n_test = len(test_labels)
    matrix = build_image_activation_matrix(
        sae, test_acts, n_test, norm_spec, device, encode_chunk=encode_chunk
    )
    x_bb = _mean_pooled_backbone_embeddings(
        test_acts, n_test, patches_per_image, norm_spec
    )
    alive = (matrix > 0).any(axis=1)
    alive_idx = np.where(alive)[0]
    if alive_idx.size == 0:
        return {
            "mono_ms_mean": 0.0,
            "mono_ms_median": 0.0,
            "mono_ms_top10": 0.0,
            "mono_ms_n_live": 0.0,
        }

    scores: List[float] = []
    k = min(top_k, n_test)
    for j in alive_idx:
        top_idx = np.argsort(-matrix[j])[:k]
        embs = x_bb[top_idx]
        scores.append(_pairwise_cosine_mean(embs))

    arr = np.array(scores, dtype=np.float64)
    top10_n = max(1, int(np.ceil(0.1 * arr.size)))
    top10 = float(np.sort(arr)[-top10_n:].mean())
    return {
        "mono_ms_mean": float(arr.mean()),
        "mono_ms_median": float(np.median(arr)),
        "mono_ms_top10": top10,
        "mono_ms_n_live": float(alive_idx.size),
    }


def compute_geo_hier(
    sae: nn.Module,
    test_acts: torch.Tensor,
    test_labels: List[str],
    norm_spec: dict,
    device: str,
    encode_chunk: int = 512,
) -> Dict[str, float]:
    """
    Per live latent: (within-group class activation similarity) -
    (across-group class activation similarity).
    """
    n_test = len(test_labels)
    matrix = build_image_activation_matrix(
        sae, test_acts, n_test, norm_spec, device, encode_chunk=encode_chunk
    )
    alive = (matrix > 0).any(axis=1)
    alive_idx = np.where(alive)[0]
    if alive_idx.size == 0:
        return {
            "geo_hier_mean": 0.0,
            "geo_hier_median": 0.0,
            "geo_hier_frac_positive": 0.0,
            "geo_hier_n_live": 0.0,
        }

    group_for_class = {c: g for g, cs in GEO_GROUPS.items() for c in cs}
    hier_scores: List[float] = []

    for j in alive_idx:
        class_vals = _class_mean_scalars(matrix[j], test_labels)
        if len(class_vals) < 2:
            continue
        within: List[float] = []
        across: List[float] = []
        classes = sorted(class_vals.keys())
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1 :]:
                sim = _scalar_cosine(class_vals[c1], class_vals[c2])
                g1, g2 = group_for_class.get(c1), group_for_class.get(c2)
                if g1 and g2 and g1 == g2:
                    within.append(sim)
                elif g1 and g2:
                    across.append(sim)
        if not within or not across:
            continue
        hier_scores.append(float(np.mean(within) - np.mean(across)))

    if not hier_scores:
        return {
            "geo_hier_mean": float("nan"),
            "geo_hier_median": float("nan"),
            "geo_hier_frac_positive": 0.0,
            "geo_hier_n_live": float(alive_idx.size),
        }

    arr = np.array(hier_scores, dtype=np.float64)
    return {
        "geo_hier_mean": float(arr.mean()),
        "geo_hier_median": float(np.median(arr)),
        "geo_hier_frac_positive": float((arr > 0).mean()),
        "geo_hier_n_live": float(alive_idx.size),
    }


@torch.no_grad()
def compute_spatial_consistency(
    sae: nn.Module,
    test_acts: torch.Tensor,
    norm_spec: dict,
    device: str,
    grid_h: int,
    grid_w: int,
    patches_per_image: int,
    run_dir: Path,
    top_heatmaps: int = 50,
    top_images: int = 20,
    encode_chunk: int = 256,
) -> Dict[str, float]:
    """
    Entropy of mean patch activation heatmaps for top activating images per latent.
    Saves top latent heatmaps under run_dir/spatial_heatmaps/.
    """
    n_images = test_acts.shape[0] // patches_per_image
    acts_cpu = preprocess_activations(test_acts.float(), norm_spec).cpu()
    dict_size = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features

    patch_cube = np.zeros((n_images, dict_size, grid_h, grid_w), dtype=np.float32)
    for img_i in range(n_images):
        start = img_i * patches_per_image
        img_patches = acts_cpu[start : start + patches_per_image]
        for cs in range(0, img_patches.shape[0], encode_chunk):
            batch = img_patches[cs : cs + encode_chunk].to(device)
            encoded, _ = sae.encode(batch)
            enc_cpu = encoded.detach().cpu().numpy()
            for local_i, gidx in enumerate(range(cs, cs + enc_cpu.shape[0])):
                if gidx >= patches_per_image:
                    break
                gy, gx = divmod(gidx, grid_w)
                patch_cube[img_i, :, gy, gx] = np.maximum(
                    patch_cube[img_i, :, gy, gx], enc_cpu[local_i]
                )

    max_per_image = patch_cube.max(axis=(2, 3))
    alive = (max_per_image > 0).any(axis=0)
    alive_idx = np.where(alive)[0]
    entropies: List[float] = []
    latent_strength = max_per_image[:, alive_idx].max(axis=0)
    top_save = alive_idx[np.argsort(-latent_strength)[:top_heatmaps]]

    heatmap_dir = run_dir / "spatial_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for j in alive_idx:
        img_scores = max_per_image[:, j]
        top_idx = np.argsort(-img_scores)[: min(top_images, n_images)]
        mean_map = patch_cube[top_idx, j].mean(axis=0)
        flat = mean_map.ravel().astype(np.float64)
        flat = flat / flat.sum() if flat.sum() > 0 else np.ones_like(flat) / flat.size
        entropies.append(float(scipy_entropy(flat)))

    for rank, j in enumerate(top_save):
        img_scores = max_per_image[:, j]
        top_idx = np.argsort(-img_scores)[: min(top_images, n_images)]
        mean_map = patch_cube[top_idx, j].mean(axis=0)
        np.save(heatmap_dir / f"latent_{int(j):05d}_rank{rank:02d}.npy", mean_map)

    arr = np.array(entropies, dtype=np.float64)
    meta = {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "patches_per_image": patches_per_image,
        "top_images": top_images,
        "n_heatmaps_saved": len(top_save),
    }
    with (heatmap_dir / "spatial_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "spatial_cons_mean": float(arr.mean()) if arr.size else float("nan"),
        "spatial_cons_std": float(arr.std()) if arr.size else float("nan"),
        "spatial_cons_n_live": float(alive_idx.size),
    }


def compute_extra_metrics(
    sae: nn.Module,
    test_acts: torch.Tensor,
    test_labels: List[str],
    norm_spec: dict,
    device: str,
    backbone_cfg: dict,
    run_dir: Path,
    metrics: set[str],
    encode_chunk: int = 512,
) -> Dict[str, Any]:
    """Run selected extra metric groups."""
    sae.eval()
    ppi = patches_per_image(backbone_cfg)
    out: Dict[str, Any] = {}

    if "dec_ortho" in metrics:
        out.update(compute_decoder_orthogonality(sae))

    need_acts = {"mono_ms", "geo_hier", "spatial"} & metrics
    if need_acts:
        n_test = len(test_labels)
        if test_acts.shape[0] != n_test * ppi:
            ppi = test_acts.shape[0] // n_test

    if "mono_ms" in metrics:
        out.update(
            compute_mono_ms(
                sae,
                test_acts,
                test_labels,
                norm_spec,
                device,
                ppi,
                encode_chunk=encode_chunk,
            )
        )

    if "geo_hier" in metrics:
        out.update(
            compute_geo_hier(
                sae, test_acts, test_labels, norm_spec, device, encode_chunk=encode_chunk
            )
        )

    if "spatial" in metrics:
        gh, gw = patch_grid_shape(backbone_cfg)
        out.update(
            compute_spatial_consistency(
                sae,
                test_acts,
                norm_spec,
                device,
                gh,
                gw,
                ppi,
                run_dir,
                encode_chunk=min(encode_chunk, 256),
            )
        )

    return out


def parse_extra_metrics_arg(raw: str) -> set[str]:
    if not raw or not raw.strip():
        return set()
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if "all" in parts:
        return {"dec_ortho", "mono_ms", "geo_hier", "spatial"}
    allowed = {"dec_ortho", "mono_ms", "geo_hier", "spatial", "ood"}
    unknown = parts - allowed
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; use dec_ortho,mono_ms,geo_hier,spatial,ood,all")
    return parts
