"""PCA / effective-rank utilities for activation preprocessing and diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

PCA_FIT_MAX_SAMPLES = 50_000


def subsample_activation_rows(
    acts: torch.Tensor,
    max_samples: int = PCA_FIT_MAX_SAMPLES,
    seed: int = 42,
) -> torch.Tensor:
    """Random row subsample for stable PCA/ZCA fitting on large patch buffers."""
    n = acts.shape[0]
    if n <= max_samples:
        return acts
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:max_samples]
    return acts[idx]


def pca_eigenvalues(acts: torch.Tensor) -> torch.Tensor:
    """Return descending covariance eigenvalues for acts [N, D]."""
    n = acts.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 activation rows for PCA")
    centered = acts.float() - acts.float().mean(dim=0)
    _, singular, _ = torch.linalg.svd(centered, full_matrices=False)
    return (singular**2) / max(n - 1, 1)


def k_for_variance_fraction(eigvals: torch.Tensor, fraction: float) -> int:
    total = float(eigvals.sum())
    if total <= 0:
        return 1
    cum = torch.cumsum(eigvals, dim=0) / total
    target = torch.tensor(fraction, dtype=cum.dtype)
    idx = int(torch.searchsorted(cum, target).item())
    return min(idx + 1, int(eigvals.numel()))


def participation_ratio(eigvals: torch.Tensor) -> float:
    s = float(eigvals.sum())
    s2 = float((eigvals**2).sum())
    if s2 <= 0:
        return float(eigvals.numel())
    return (s * s) / s2


def variance_threshold_ks(
    eigvals: torch.Tensor,
    thresholds: Iterable[float] = (0.50, 0.80, 0.90, 0.95, 0.99),
) -> Dict[str, int]:
    return {
        f"k_{int(threshold * 100)}": k_for_variance_fraction(eigvals, threshold)
        for threshold in thresholds
    }


def summarize_effective_rank(acts: torch.Tensor) -> Dict[str, float | int]:
    """Effective-rank summary for a matrix of shape [N, hidden_dim]."""
    eigvals = pca_eigenvalues(acts)
    out: Dict[str, float | int] = {
        "hidden_dim": int(acts.shape[1]),
        "n_samples": int(acts.shape[0]),
        "participation_ratio": participation_ratio(eigvals),
    }
    out.update(variance_threshold_ks(eigvals))
    return out


def fit_pca_projection(
    acts: torch.Tensor,
    variance_threshold: float = 0.95,
    max_fit_samples: int = PCA_FIT_MAX_SAMPLES,
    seed: int = 42,
) -> Dict[str, torch.Tensor | int | float]:
    fit_acts = subsample_activation_rows(acts, max_samples=max_fit_samples, seed=seed)
    mean = fit_acts.float().mean(dim=0)
    centered = fit_acts.float() - mean
    n = centered.shape[0]
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    eigvals = (singular**2) / max(n - 1, 1)
    k = k_for_variance_fraction(eigvals, variance_threshold)
    components = vh[:k].clone()
    return {
        "act_mean": mean.cpu(),
        "pca_components": components.cpu(),
        "pca_k_95": int(k),
        "projected_dim": int(k),
        "pca_variance_threshold": float(variance_threshold),
    }


def fit_zca_whitening(
    acts: torch.Tensor,
    eps: float = 1e-3,
    max_fit_samples: int = PCA_FIT_MAX_SAMPLES,
    seed: int = 42,
) -> Dict[str, torch.Tensor | float]:
    fit_acts = subsample_activation_rows(acts, max_samples=max_fit_samples, seed=seed)
    mean = fit_acts.float().mean(dim=0)
    centered = fit_acts.float() - mean
    n, _ = centered.shape
    cov = (centered.T @ centered) / max(n - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    inv_sqrt = 1.0 / torch.sqrt(eigvals.clamp_min(0.0) + eps)
    whitening = eigvecs @ torch.diag(inv_sqrt) @ eigvecs.T
    return {
        "act_mean": mean.cpu(),
        "zca_matrix": whitening.cpu(),
        "zca_eps": float(eps),
    }


def apply_pca_projection(
    activations: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    mean = mean.to(device=activations.device, dtype=activations.dtype)
    components = components.to(device=activations.device, dtype=activations.dtype)
    return (activations - mean) @ components.T


def apply_zca_whitening(
    activations: torch.Tensor,
    mean: torch.Tensor,
    whitening: torch.Tensor,
) -> torch.Tensor:
    mean = mean.to(device=activations.device, dtype=activations.dtype)
    whitening = whitening.to(device=activations.device, dtype=activations.dtype)
    return (activations - mean) @ whitening.T
