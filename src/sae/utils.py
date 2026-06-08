"""Shared SAE utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def normalize_decoder_columns(decoder: nn.Linear) -> None:
    weight = decoder.weight.data
    norms = weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
    decoder.weight.data = weight / norms


def fold_norm_scalar_into_sae(
    sae: nn.Module,
    norm_scalar: float,
) -> None:
    """Fold activation normalization scalar into SAE weights (inference without norm)."""
    if norm_scalar == 1.0:
        return
    sae.encoder.weight.data = sae.encoder.weight.data / norm_scalar
    if sae.encoder.bias is not None:
        sae.encoder.bias.data = sae.encoder.bias.data / norm_scalar
    sae.decoder.weight.data = sae.decoder.weight.data * norm_scalar
    if hasattr(sae, "b_dec") and sae.b_dec is not None:
        sae.b_dec.data = sae.b_dec.data * norm_scalar


def compute_norm_scalar(activations: torch.Tensor) -> float:
    mean_sq_norm = activations.pow(2).sum(dim=-1).mean().item()
    return float(mean_sq_norm ** 0.5) if mean_sq_norm > 0 else 1.0


def normalize_activations(activations: torch.Tensor, scalar: float) -> torch.Tensor:
    if scalar <= 0 or scalar == 1.0:
        return activations
    return activations / scalar


def sae_inference_norm_scalar(checkpoint_payload: dict) -> float:
    """Backward-compatible scalar for legacy scripts."""
    from src.sae.activation_norm import inference_norm_scalar

    return inference_norm_scalar(checkpoint_payload)
