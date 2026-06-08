"""
HiRISE sparse probing — SAEBench sparse probing adapted for vision.

SAEBench: https://arxiv.org/html/2503.09532 — linear probe on top SAE latents per concept.
Here: multi-class HiRISE v3.2 landform labels; compare backbone vs SAE latent probes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.hirise import MIN_PER_CLASS_PROBE_REPORT
from src.sae.activation_norm import preprocess_activations

# HiRISE v3.2 landmarks_map classmap (ID -> name)
HIRISE_LANDFORM_NAMES: Dict[str, str] = {
    "1": "crater",
    "2": "dark dune",
    "3": "slope streak",
    "4": "bright dune",
    "5": "impact ejecta",
    "6": "swiss cheese",
    "7": "spider",
}

BACKGROUND_LABELS = frozenset({"0", "other", "background", ""})


def is_named_landform(label: str) -> bool:
    """True for HiRISE classes 1–7 (excludes class 0 / other)."""
    s = str(label).strip().lower()
    return s not in BACKGROUND_LABELS


def label_display_name(label: str) -> str:
    s = str(label).strip()
    return HIRISE_LANDFORM_NAMES.get(s, s)


def _pool_image_activations(
    patch_acts: torch.Tensor,
    n_images: int,
    patches_per_image: int,
) -> torch.Tensor:
    """Mean-pool patch tokens per image -> [n_images, d_in]."""
    n_patches = patch_acts.shape[0]
    if n_patches != n_images * patches_per_image:
        patches_per_image = n_patches // n_images
    return patch_acts[: n_images * patches_per_image].reshape(
        n_images, patches_per_image, -1
    ).mean(dim=1)


def _train_probe_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    max_iter: int = 2000,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)

    per_class = f1_score(
        y_test, pred, average=None, zero_division=0, labels=np.arange(len(class_names))
    )
    per_class_named = {
        label_display_name(class_names[i]): float(per_class[i])
        for i in range(len(class_names))
    }

    return {
        "macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "per_class": per_class_named,
    }


def _macro_f1_well_represented(
    per_class: Dict[str, float],
    test_class_counts: Dict[str, int],
    min_count: int = MIN_PER_CLASS_PROBE_REPORT,
) -> float:
    """Macro-F1 over test classes with at least min_count images (paper reporting subset)."""
    vals = []
    for cls_id, count in test_class_counts.items():
        if count < min_count:
            continue
        name = label_display_name(cls_id)
        if name in per_class:
            vals.append(per_class[name])
    return float(np.mean(vals)) if vals else float("nan")


def _filter_landform_images(
    labels: List[str],
    landforms_only: bool,
) -> np.ndarray:
    """Indices of images to include in the probe."""
    if not landforms_only:
        return np.arange(len(labels))
    return np.array([i for i, lab in enumerate(labels) if is_named_landform(lab)])


@torch.no_grad()
def _encode_image_latent_means(
    sae: nn.Module,
    acts_cpu: torch.Tensor,
    n_images: int,
    patches_per_image: int,
    device: str,
    encode_chunk: int = 512,
) -> np.ndarray:
    """Mean-pool SAE latents per image without materializing all patches at once."""
    latent_rows = []
    for img_i in range(n_images):
        start = img_i * patches_per_image
        end = start + patches_per_image
        img_patches = acts_cpu[start:end]
        encoded_parts = []
        for chunk_start in range(0, img_patches.shape[0], encode_chunk):
            batch = img_patches[chunk_start : chunk_start + encode_chunk].to(device)
            encoded, _ = sae.encode(batch)
            encoded_parts.append(encoded.cpu())
        latent_rows.append(torch.cat(encoded_parts, dim=0).mean(dim=0))
    return torch.stack(latent_rows, dim=0).numpy()


@torch.no_grad()
def _encode_pooled_features(
    sae: nn.Module,
    backbone_acts: torch.Tensor,
    n_images: int,
    norm_spec: dict,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-pooled backbone and SAE latent features per image."""
    patches_per_image = backbone_acts.shape[0] // n_images
    acts_cpu = preprocess_activations(backbone_acts.float(), norm_spec).cpu()
    x_backbone = _pool_image_activations(acts_cpu, n_images, patches_per_image).numpy()
    x_latent = _encode_image_latent_means(
        sae, acts_cpu, n_images, patches_per_image, device
    )
    return x_backbone, x_latent


@torch.no_grad()
def compute_hirise_sparse_probing(
    sae: nn.Module,
    backbone_acts: Optional[torch.Tensor] = None,
    n_images: Optional[int] = None,
    labels: Optional[List[str]] = None,
    norm_spec: dict | None = None,
    norm_scalar: float = 1.0,
    top_k_latents: int = 40,
    test_size: float = 0.2,
    seed: int = 42,
    landforms_only: bool = False,
    train_backbone_acts: Optional[torch.Tensor] = None,
    train_labels: Optional[List[str]] = None,
    test_backbone_acts: Optional[torch.Tensor] = None,
    test_labels: Optional[List[str]] = None,
    official_split: bool = False,
    test_class_counts: Optional[Dict[str, int]] = None,
    min_per_class_report: int = MIN_PER_CLASS_PROBE_REPORT,
) -> Dict[str, Any]:
    """
    Linear probes on mean-pooled patch features per image.

    landforms_only: drop class 0 (other); probe only named geomorphology classes 1–7.

    official_split: fit on train_backbone_acts / train_labels, evaluate on
    test_backbone_acts / test_labels (HiRISE v3.2 creator test split, originals only).

    Otherwise: random holdout on the provided activation tensor (legacy ad-hoc eval).
    """
    if norm_spec is None:
        norm_spec = {"preprocess_mode": "scalar", "norm_scalar": norm_scalar}
    device = next(sae.parameters()).device

    if official_split:
        if (
            train_backbone_acts is None
            or test_backbone_acts is None
            or train_labels is None
            or test_labels is None
        ):
            raise ValueError("official_split requires train and test activations + labels")
        n_train = len(train_labels)
        n_test = len(test_labels)
        x_bb_train, x_lat_train = _encode_pooled_features(
            sae, train_backbone_acts, n_train, norm_spec, device
        )
        x_bb_test, x_lat_test = _encode_pooled_features(
            sae, test_backbone_acts, n_test, norm_spec, device
        )
        le = LabelEncoder()
        le.fit(train_labels + test_labels)
        y_train = le.transform(train_labels)
        y_test = le.transform(test_labels)
        class_names = [str(c) for c in le.classes_]
        n_classes = len(class_names)

        if n_classes < 2 or n_train < 10 or n_test < 10:
            return _empty_probe_result(landforms_only, official_split=True, n_test=n_test)

        scores_bb = _train_probe_scores(
            x_bb_train, y_train, x_bb_test, y_test, class_names
        )
        scores_sae = _train_probe_scores(
            x_lat_train, y_train, x_lat_test, y_test, class_names
        )
        firing_rate = (x_lat_train > 0).mean(axis=0)
        k = min(top_k_latents, x_lat_train.shape[1])
        top_idx = np.argsort(-firing_rate)[:k]
        scores_topk = _train_probe_scores(
            x_lat_train[:, top_idx],
            y_train,
            x_lat_test[:, top_idx],
            y_test,
            class_names,
        )
        counts = test_class_counts or {}
        return _build_probe_result(
            scores_bb,
            scores_sae,
            scores_topk,
            n_classes=n_classes,
            n_eval_images=n_test,
            n_train_images=n_train,
            landforms_only=landforms_only,
            official_split=True,
            top_k=k,
            test_class_counts=counts,
            min_per_class_report=min_per_class_report,
        )

    if backbone_acts is None or n_images is None or labels is None:
        raise ValueError("Non-official probing requires backbone_acts, n_images, and labels")

    patches_per_image = backbone_acts.shape[0] // n_images
    img_idx = _filter_landform_images(labels, landforms_only)
    if len(img_idx) < 10:
        return _empty_probe_result(landforms_only, official_split=False, n_test=len(img_idx))

    filtered_labels = [labels[i] for i in img_idx]
    n_filt = len(img_idx)
    acts_cpu = preprocess_activations(backbone_acts.float(), norm_spec).cpu()

    def subset_patches(full: torch.Tensor) -> np.ndarray:
        rows = []
        for i in img_idx:
            start = int(i) * patches_per_image
            end = start + patches_per_image
            rows.append(full[start:end].mean(dim=0))
        return torch.stack(rows, dim=0).numpy()

    x_backbone = subset_patches(acts_cpu)
    latent_rows = []
    for i in img_idx:
        start = int(i) * patches_per_image
        end = start + patches_per_image
        img_patches = acts_cpu[start:end]
        encoded_parts = []
        for chunk_start in range(0, img_patches.shape[0], 512):
            batch = img_patches[chunk_start : chunk_start + 512].to(device)
            encoded, _ = sae.encode(batch)
            encoded_parts.append(encoded.cpu())
        latent_rows.append(torch.cat(encoded_parts, dim=0).mean(dim=0))
    x_latent = torch.stack(latent_rows, dim=0).numpy()

    le = LabelEncoder()
    y = le.fit_transform(filtered_labels)
    class_names = [str(c) for c in le.classes_]
    n_classes = len(class_names)

    if n_classes < 2:
        return _empty_probe_result(landforms_only, official_split=False, n_test=n_filt)

    split_kwargs: Dict[str, Any] = {"test_size": test_size, "random_state": seed}
    _, counts = np.unique(y, return_counts=True)
    if counts.min() >= 2 and n_filt >= 5 * n_classes:
        split_kwargs["stratify"] = y
    train_idx, test_idx = train_test_split(np.arange(n_filt), **split_kwargs)

    scores_bb = _train_probe_scores(
        x_backbone[train_idx],
        y[train_idx],
        x_backbone[test_idx],
        y[test_idx],
        class_names,
    )
    scores_sae = _train_probe_scores(
        x_latent[train_idx],
        y[train_idx],
        x_latent[test_idx],
        y[test_idx],
        class_names,
    )
    train_latent = x_latent[train_idx]
    firing_rate = (train_latent > 0).mean(axis=0)
    k = min(top_k_latents, x_latent.shape[1])
    top_idx = np.argsort(-firing_rate)[:k]
    scores_topk = _train_probe_scores(
        x_latent[train_idx][:, top_idx],
        y[train_idx],
        x_latent[test_idx][:, top_idx],
        y[test_idx],
        class_names,
    )

    return _build_probe_result(
        scores_bb,
        scores_sae,
        scores_topk,
        n_classes=n_classes,
        n_eval_images=n_filt,
        n_train_images=int(len(train_idx)),
        landforms_only=landforms_only,
        official_split=False,
        top_k=k,
        test_class_counts={},
        min_per_class_report=min_per_class_report,
    )


def _empty_probe_result(
    landforms_only: bool,
    official_split: bool,
    n_test: int,
) -> Dict[str, Any]:
    return {
        "sparse_probe_f1_backbone": float("nan"),
        "sparse_probe_f1_sae_latents": float("nan"),
        "sparse_probe_f1_sae_topk": float("nan"),
        "sparse_probe_f1_backbone_weighted": float("nan"),
        "sparse_probe_f1_sae_latents_weighted": float("nan"),
        "sparse_probe_f1_sae_topk_weighted": float("nan"),
        "sparse_probe_f1_backbone_core_classes": float("nan"),
        "sparse_probe_f1_sae_latents_core_classes": float("nan"),
        "sparse_probe_num_classes": 0.0,
        "sparse_probe_num_images": float(n_test),
        "sparse_probe_num_train_images": 0.0,
        "sparse_probe_split": "official_test" if official_split else "adhoc_holdout",
        "sparse_probe_landforms_only": landforms_only,
        "sparse_probe_per_class_f1_backbone": {},
        "sparse_probe_per_class_f1_sae": {},
    }


def _build_probe_result(
    scores_bb: Dict[str, Any],
    scores_sae: Dict[str, Any],
    scores_topk: Dict[str, Any],
    n_classes: int,
    n_eval_images: int,
    n_train_images: int,
    landforms_only: bool,
    official_split: bool,
    top_k: int,
    test_class_counts: Dict[str, int],
    min_per_class_report: int,
) -> Dict[str, Any]:
    return {
        "sparse_probe_f1_backbone": scores_bb["macro"],
        "sparse_probe_f1_sae_latents": scores_sae["macro"],
        "sparse_probe_f1_sae_topk": scores_topk["macro"],
        "sparse_probe_f1_backbone_weighted": scores_bb["weighted"],
        "sparse_probe_f1_sae_latents_weighted": scores_sae["weighted"],
        "sparse_probe_f1_sae_topk_weighted": scores_topk["weighted"],
        "sparse_probe_f1_backbone_core_classes": _macro_f1_well_represented(
            scores_bb["per_class"], test_class_counts, min_per_class_report
        ),
        "sparse_probe_f1_sae_latents_core_classes": _macro_f1_well_represented(
            scores_sae["per_class"], test_class_counts, min_per_class_report
        ),
        "sparse_probe_num_classes": float(n_classes),
        "sparse_probe_num_images": float(n_eval_images),
        "sparse_probe_num_train_images": float(n_train_images),
        "sparse_probe_top_k": float(top_k),
        "sparse_probe_split": "official_test" if official_split else "adhoc_holdout",
        "sparse_probe_landforms_only": landforms_only,
        "sparse_probe_test_class_counts": test_class_counts,
        "sparse_probe_min_per_class_report": float(min_per_class_report),
        "sparse_probe_per_class_f1_backbone": scores_bb["per_class"],
        "sparse_probe_per_class_f1_sae": scores_sae["per_class"],
        "sparse_probe_per_class_f1_sae_topk": scores_topk["per_class"],
    }
