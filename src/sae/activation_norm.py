"""Activation preprocessing for SAE training and eval."""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.extract.activations import ActivationBuffer
from src.sae.effective_rank import (
    apply_pca_projection,
    apply_zca_whitening,
    fit_pca_projection,
    fit_zca_whitening,
)
from src.sae.utils import compute_norm_scalar

EXPERIMENTAL_PREPROCESS_MODES = frozenset({"pca_proj", "zca_whiten"})


def fit_activation_norm(
    buffer: ActivationBuffer,
    mode: str = "scalar",
    center: bool = True,
) -> Dict[str, Any]:
    """
    Fit normalization stats on the activation buffer (first fill).

    scalar: divide by global RMS norm (legacy default).
    per_dim: (x - mean) / std per channel; recommended for anisotropic backbones.
    pca_proj: project to the PCA subspace explaining 95% train-buffer variance.
    zca_whiten: ZCA whitening with eps=1e-3 on the train activation buffer.
    """
    acts = buffer.storage.float()
    if mode == "per_dim":
        mean = acts.mean(dim=0) if center else torch.zeros(acts.shape[1])
        std = acts.std(dim=0).clamp_min(1e-8)
        return {
            "preprocess_mode": "per_dim",
            "norm_scalar": 1.0,
            "center_activations": center,
            "act_mean": mean.cpu(),
            "act_std": std.cpu(),
        }
    if mode == "pca_proj":
        spec = fit_pca_projection(acts, variance_threshold=0.95)
        spec.update(
            {
                "preprocess_mode": "pca_proj",
                "norm_scalar": 1.0,
                "center_activations": True,
                "act_std": None,
            }
        )
        return spec
    if mode == "zca_whiten":
        spec = fit_zca_whitening(acts, eps=1e-3)
        spec.update(
            {
                "preprocess_mode": "zca_whiten",
                "norm_scalar": 1.0,
                "center_activations": True,
                "act_std": None,
            }
        )
        return spec
    if mode != "scalar":
        raise ValueError(f"Unknown preprocess mode: {mode!r}")
    return {
        "preprocess_mode": "scalar",
        "norm_scalar": compute_norm_scalar(acts),
        "center_activations": False,
        "act_mean": None,
        "act_std": None,
    }


def activation_norm_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct norm spec from a checkpoint payload."""
    mode = payload.get("preprocess_mode", "scalar")
    spec: Dict[str, Any] = {
        "preprocess_mode": mode,
        "norm_scalar": float(payload.get("norm_scalar", 1.0)),
        "center_activations": bool(payload.get("center_activations", False)),
        "act_mean": payload.get("act_mean"),
        "act_std": payload.get("act_std"),
        "pca_components": payload.get("pca_components"),
        "pca_k_95": payload.get("pca_k_95"),
        "projected_dim": payload.get("projected_dim"),
        "pca_variance_threshold": payload.get("pca_variance_threshold"),
        "zca_matrix": payload.get("zca_matrix"),
        "zca_eps": payload.get("zca_eps"),
    }
    if mode == "per_dim":
        if spec["act_mean"] is None or spec["act_std"] is None:
            raise ValueError("per_dim checkpoint missing act_mean/act_std")
        spec["act_mean"] = torch.as_tensor(spec["act_mean"], dtype=torch.float32)
        spec["act_std"] = torch.as_tensor(spec["act_std"], dtype=torch.float32)
    elif mode == "pca_proj":
        if spec["act_mean"] is None or spec["pca_components"] is None:
            raise ValueError("pca_proj checkpoint missing act_mean/pca_components")
        spec["act_mean"] = torch.as_tensor(spec["act_mean"], dtype=torch.float32)
        spec["pca_components"] = torch.as_tensor(
            spec["pca_components"], dtype=torch.float32
        )
        spec["projected_dim"] = int(
            spec.get("projected_dim") or spec["pca_components"].shape[0]
        )
        spec["pca_k_95"] = int(spec.get("pca_k_95") or spec["projected_dim"])
    elif mode == "zca_whiten":
        if spec["act_mean"] is None or spec["zca_matrix"] is None:
            raise ValueError("zca_whiten checkpoint missing act_mean/zca_matrix")
        spec["act_mean"] = torch.as_tensor(spec["act_mean"], dtype=torch.float32)
        spec["zca_matrix"] = torch.as_tensor(spec["zca_matrix"], dtype=torch.float32)
    return spec


def preprocess_activations(
    activations: torch.Tensor,
    norm_spec: Dict[str, Any],
) -> torch.Tensor:
    """Apply checkpoint-consistent preprocessing before SAE encode."""
    mode = norm_spec.get("preprocess_mode", "scalar")
    if mode == "per_dim":
        mean = norm_spec["act_mean"].to(
            device=activations.device, dtype=activations.dtype
        )
        std = norm_spec["act_std"].to(
            device=activations.device, dtype=activations.dtype
        )
        return (activations - mean) / std
    if mode == "pca_proj":
        return apply_pca_projection(
            activations,
            norm_spec["act_mean"],
            norm_spec["pca_components"],
        )
    if mode == "zca_whiten":
        return apply_zca_whitening(
            activations,
            norm_spec["act_mean"],
            norm_spec["zca_matrix"],
        )
    scalar = float(norm_spec.get("norm_scalar", 1.0))
    if scalar <= 0 or scalar == 1.0:
        return activations
    return activations / scalar


def inference_norm_scalar(payload: Dict[str, Any]) -> float:
    """
    Scalar applied at eval when weights were saved with scalar folding.

    per_dim / pca_proj / zca_whiten checkpoints preprocess at runtime.
    """
    if payload.get("preprocess_mode") in {"per_dim", "pca_proj", "zca_whiten"}:
        return 1.0
    if payload.get("norm_folded", True):
        return 1.0
    return float(payload.get("norm_scalar", 1.0))


def norm_spec_for_eval(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Full spec for eval paths (scalar fold + optional per-dim stats)."""
    spec = activation_norm_from_payload(payload)
    if payload.get("preprocess_mode", "scalar") == "scalar" and payload.get(
        "norm_folded", True
    ):
        spec["norm_scalar"] = 1.0
    return spec


def sae_input_dim(hidden_dim: int, norm_spec: Dict[str, Any]) -> int:
    """Effective SAE input dimension after preprocessing."""
    if norm_spec.get("preprocess_mode") == "pca_proj":
        return int(norm_spec.get("projected_dim") or norm_spec.get("pca_k_95") or hidden_dim)
    return hidden_dim
