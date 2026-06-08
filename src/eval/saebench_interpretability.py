"""
SAEBench-style interpretability metrics (Priority 1).

- Feature suppression / causal intervention (Stevens et al. arXiv:2502.06755)
- Monosemanticity (label purity on top-k activating images)
- TCAV-style probe–decoder alignment (Lemma 4 geometric check)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from src.eval.monosemanticity import label_purity_topk
from src.eval.saebench_sparse_probing import (
    _encode_pooled_features,
    label_display_name,
)
from src.sae.activation_norm import preprocess_activations


def fit_latent_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    max_iter: int = 2000,
) -> Tuple[LogisticRegression, StandardScaler, float]:
    """Train multinomial logistic probe on SAE latents; return baseline macro F1."""
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)
    baseline_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return clf, scaler, baseline_f1


@torch.no_grad()
def _pooled_latents_error_reinjection(
    sae: nn.Module,
    backbone_acts: torch.Tensor,
    n_images: int,
    norm_spec: dict,
    device: str,
    ablate_idx: Optional[int] = None,
    encode_chunk: int = 512,
) -> np.ndarray:
    """
    Mean-pooled SAE latents per image with optional single-feature ablation.

    Stevens-style intervention: x' = e + decode(f(x)'), e = x - decode(f(x)),
    then re-encode x' for probe features.
    """
    patches_per_image = backbone_acts.shape[0] // n_images
    acts_cpu = preprocess_activations(backbone_acts.float(), norm_spec).cpu()
    rows: List[torch.Tensor] = []

    for img_i in range(n_images):
        start = img_i * patches_per_image
        end = start + patches_per_image
        img_patches = acts_cpu[start:end]
        latent_parts: List[torch.Tensor] = []
        for cs in range(0, img_patches.shape[0], encode_chunk):
            batch = img_patches[cs : cs + encode_chunk].to(device)
            acts, _ = sae.encode(batch)
            recon = sae.decode(acts)
            err = batch - recon
            if ablate_idx is not None:
                acts = acts.clone()
                acts[:, ablate_idx] = 0.0
            x_prime = err + sae.decode(acts)
            acts_new, _ = sae.encode(x_prime)
            latent_parts.append(acts_new.cpu())
        rows.append(torch.cat(latent_parts, dim=0).mean(dim=0))

    return torch.stack(rows, dim=0).numpy()


def label_purity_topk_strict(
    matrix: np.ndarray,
    labels: List[str],
    top_k: int = 20,
) -> np.ndarray:
    """
    Monosemanticity: max class count / k among the top-k images by activation
    (includes zero-activation slots in the denominator).
    """
    n_features, n_images = matrix.shape
    k = min(top_k, n_images)
    purities = np.zeros(n_features, dtype=np.float64)
    for i in range(n_features):
        if k == 0:
            continue
        top_idx = np.argsort(-matrix[i])[:k]
        top_labels = [labels[j] for j in top_idx]
        _, counts = np.unique(top_labels, return_counts=True)
        purities[i] = counts.max() / k
    return purities


@torch.no_grad()
def build_image_activation_matrix(
    sae: nn.Module,
    backbone_acts: torch.Tensor,
    n_images: int,
    norm_spec: dict,
    device: str,
    encode_chunk: int = 512,
) -> np.ndarray:
    """Per-latent max activation across patches within each image."""
    patches_per_image = backbone_acts.shape[0] // n_images
    acts_cpu = preprocess_activations(backbone_acts.float(), norm_spec).cpu()
    dict_size = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features
    matrix = np.zeros((dict_size, n_images), dtype=np.float64)

    for img_i in range(n_images):
        start = img_i * patches_per_image
        end = start + patches_per_image
        img_patches = acts_cpu[start:end]
        max_lat = torch.zeros(dict_size)
        for cs in range(0, img_patches.shape[0], encode_chunk):
            batch = img_patches[cs : cs + encode_chunk].to(device)
            encoded, _ = sae.encode(batch)
            max_lat = torch.maximum(max_lat, encoded.max(dim=0).values.detach().cpu())
        matrix[:, img_i] = max_lat.numpy()

    return matrix


def compute_tcav_probe_decoder_alignment(
    sae: nn.Module,
    clf: LogisticRegression,
) -> Dict[str, Any]:
    """
    For each landform class c, project probe coefficients into activation space
    via the decoder (t_c = W_dec @ coef[c]) and measure max cosine similarity
    to each decoder column (Lemma 4 geometric correspondence).
    """
    w_dec = sae.decoder.weight.detach().cpu().float()  # (d_in, n_latents)
    coef = torch.tensor(clf.coef_, dtype=torch.float32)  # (n_classes, n_latents)
    d_in, n_latents = w_dec.shape
    w_dec_norm = w_dec / w_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)

    max_per_class: List[float] = []
    all_sims: List[float] = []
    per_class: Dict[str, float] = {}

    for c in range(coef.shape[0]):
        t_c = w_dec @ coef[c]
        t_norm = t_c / t_c.norm().clamp_min(1e-8)
        sims = (t_norm.unsqueeze(1) * w_dec_norm).sum(dim=0).numpy()
        max_sim = float(sims.max())
        max_per_class.append(max_sim)
        all_sims.extend(sims.tolist())
        if c < len(clf.classes_):
            per_class[label_display_name(str(clf.classes_[c]))] = max_sim

    arr = np.array(max_per_class, dtype=np.float64)
    all_arr = np.array(all_sims, dtype=np.float64)
    return {
        "tcav_mean_max_cos_per_class": float(arr.mean()) if arr.size else float("nan"),
        "tcav_median_max_cos_per_class": float(np.median(arr)) if arr.size else float("nan"),
        "tcav_max_cos_any_class": float(arr.max()) if arr.size else float("nan"),
        "tcav_mean_cos_all_pairs": float(all_arr.mean()) if all_arr.size else float("nan"),
        "tcav_max_cos_per_class": per_class,
    }


def compute_causal_intervention_metrics(
    sae: nn.Module,
    train_acts: torch.Tensor,
    train_labels: List[str],
    test_acts: torch.Tensor,
    test_labels: List[str],
    norm_spec: dict,
    device: str,
    run_dir: Path,
    encode_chunk: int = 512,
) -> Dict[str, Any]:
    """Per-latent F1 drop under zero-ablation with error reinjection."""
    n_train = len(train_labels)
    n_test = len(test_labels)
    _, x_train = _encode_pooled_features(sae, train_acts, n_train, norm_spec, device)
    _, x_test_baseline = _encode_pooled_features(
        sae, test_acts, n_test, norm_spec, device
    )

    le = LabelEncoder()
    le.fit(train_labels + test_labels)
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)

    clf, scaler, baseline_f1 = fit_latent_probe(
        x_train, y_train, x_test_baseline, y_test, [str(c) for c in le.classes_]
    )

    dict_size = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features
    firing = (x_test_baseline > 0).any(axis=0)
    alive_idx = np.where(firing)[0]
    f1_drops = np.zeros(dict_size, dtype=np.float64)

    for j in tqdm(alive_idx, desc="Causal ablation", unit="latent", leave=False):
        x_abl = _pooled_latents_error_reinjection(
            sae, test_acts, n_test, norm_spec, device, ablate_idx=int(j), encode_chunk=encode_chunk
        )
        x_abl_s = scaler.transform(x_abl)
        pred = clf.predict(x_abl_s)
        ablated_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
        f1_drops[j] = baseline_f1 - ablated_f1

    np.save(run_dir / "causal_f1_drop.npy", f1_drops)
    alive_drops = f1_drops[alive_idx] if alive_idx.size else np.array([])
    summary = {
        "causal_baseline_f1": baseline_f1,
        "causal_mean_f1_drop": float(alive_drops.mean()) if alive_drops.size else 0.0,
        "causal_median_f1_drop": float(np.median(alive_drops)) if alive_drops.size else 0.0,
        "causal_max_f1_drop": float(alive_drops.max()) if alive_drops.size else 0.0,
        "causal_n_latents_alive_test": float(alive_idx.size),
    }
    with (run_dir / "causal_intervention_summary.json").open("w", encoding="utf-8") as f:
        json.dump({**summary, "f1_drop_alive_indices": alive_idx.tolist()}, f, indent=2)
    return summary


def compute_monosemanticity_metrics(
    sae: nn.Module,
    test_acts: torch.Tensor,
    test_labels: List[str],
    norm_spec: dict,
    device: str,
    run_dir: Path,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Label purity on official test images.

    Note: proxy_label_purity_top20 (from feature_activation_matrix on 500 ad-hoc
    eval images) uses label_purity_topk which excludes zero-activation slots from
    the top-k set. This metric uses the strict top-k definition and official
    test landform labels.
    """
    n_test = len(test_labels)
    matrix = build_image_activation_matrix(
        sae, test_acts, n_test, norm_spec, device
    )
    np.save(run_dir / "official_test_activation_matrix.npy", matrix)

    purities_legacy = label_purity_topk(matrix, test_labels, top_k=top_k)
    purities_strict = label_purity_topk_strict(matrix, test_labels, top_k=top_k)
    alive = (matrix > 0).any(axis=1)

    np.save(run_dir / "mono_purity_top20_official.npy", purities_strict)

    summary = {
        "mono_mean_purity_top20_official": float(purities_strict[alive].mean())
        if alive.any()
        else 0.0,
        "mono_median_purity_top20_official": float(np.median(purities_strict[alive]))
        if alive.any()
        else 0.0,
        "mono_mean_purity_top20_legacy_denom": float(purities_legacy[alive].mean())
        if alive.any()
        else 0.0,
        "mono_fraction_high_purity_top20": float((purities_strict[alive] >= 0.8).mean())
        if alive.any()
        else 0.0,
        "mono_top_k": float(top_k),
        "mono_n_test_images": float(n_test),
    }
    with (run_dir / "monosemanticity_official_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def compute_interpretability_metrics(
    sae: nn.Module,
    train_acts: torch.Tensor,
    train_labels: List[str],
    test_acts: torch.Tensor,
    test_labels: List[str],
    norm_spec: dict,
    device: str,
    run_dir: Path,
    metrics: set[str],
    encode_chunk: int = 512,
) -> Dict[str, Any]:
    """Run selected interpretability metric groups."""
    sae.eval()
    out: Dict[str, Any] = {}
    need_probe = "causal" in metrics or "tcav" in metrics

    clf = None
    if need_probe:
        n_train = len(train_labels)
        _, x_train_latent = _encode_pooled_features(
            sae, train_acts, n_train, norm_spec, device
        )
        le = LabelEncoder()
        le.fit(train_labels + test_labels)
        y_train = le.transform(train_labels)
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train_latent)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(x_train_s, y_train)

    if "causal" in metrics:
        out.update(
            compute_causal_intervention_metrics(
                sae,
                train_acts,
                train_labels,
                test_acts,
                test_labels,
                norm_spec,
                device,
                run_dir,
                encode_chunk=encode_chunk,
            )
        )

    if "mono" in metrics:
        out.update(
            compute_monosemanticity_metrics(
                sae, test_acts, test_labels, norm_spec, device, run_dir
            )
        )

    if "tcav" in metrics and clf is not None:
        tcav = compute_tcav_probe_decoder_alignment(sae, clf)
        per_class = tcav.pop("tcav_max_cos_per_class", {})
        out.update(tcav)
        with (run_dir / "tcav_probe_decoder.json").open("w", encoding="utf-8") as f:
            json.dump({**tcav, "max_cos_per_class": per_class}, f, indent=2)

    return out


def parse_metrics_arg(raw: str) -> set[str]:
    if not raw or not raw.strip():
        return set()
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if "all" in parts:
        return {"causal", "mono", "tcav"}
    allowed = {"causal", "mono", "tcav"}
    unknown = parts - allowed
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; use causal, mono, tcav, all")
    return parts
