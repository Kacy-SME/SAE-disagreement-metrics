"""Post-training SAE evaluation metrics."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.backbones.base import BackboneAdapter
from src.extract.activations import extract_patch_activations
from src.sae.activation_norm import preprocess_activations


@torch.no_grad()
def reconstruction_mse(sae: nn.Module, x: torch.Tensor) -> float:
    if hasattr(sae, "nested_reconstructions"):
        _, acts, _ = sae.encode(x) if hasattr(sae, "encode") else (None, None, None)
        recons = sae.nested_reconstructions(x, acts)
        recon = recons[-1]
    else:
        recon, _, _ = sae(x)
    return float(F.mse_loss(recon, x).item())


@torch.no_grad()
def evaluate_sae_on_activations(
    sae: nn.Module,
    activations: torch.Tensor,
    norm_spec: dict,
    device: str,
    batch_size: int = 2048,
    activations_cls_zero: torch.Tensor | None = None,
) -> Dict[str, float]:
    acts = preprocess_activations(activations, norm_spec).to(device)
    acts_cls_zero = None
    if activations_cls_zero is not None:
        acts_cls_zero = preprocess_activations(activations_cls_zero, norm_spec).to(device)
    n = acts.shape[0]

    h_with_sae = []
    h_zero_ablate = []
    h_zero_cls_proxy = []
    h_original = []
    l0_vals = []
    feature_max = None
    n_features = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features

    batch_starts = list(range(0, n, batch_size))
    for start in tqdm(batch_starts, desc="Eval SAE metrics", unit="batch", leave=False):
        batch = acts[start : start + batch_size]
        encoded_acts, _ = sae.encode(batch)

        if hasattr(sae, "nested_reconstructions"):
            recons = sae.nested_reconstructions(batch, encoded_acts)
            recon = recons[-1]
        else:
            recon = sae.decode(encoded_acts)

        zero_recon = sae.decode(torch.zeros_like(encoded_acts))
        dense_recon = sae.decode(F.relu(sae.encoder(batch - sae.b_dec)))

        h_with_sae.append(F.mse_loss(recon, batch).item())
        h_zero_ablate.append(F.mse_loss(zero_recon, batch).item())
        h_original.append(F.mse_loss(dense_recon, batch).item())
        if acts_cls_zero is not None:
            batch_cls = acts_cls_zero[start : start + batch_size]
            h_zero_cls_proxy.append(F.mse_loss(batch_cls, batch).item())
        l0_vals.append((encoded_acts > 0).float().sum(dim=-1).mean().item())

    h_with = float(np.mean(h_with_sae))
    h_zero = float(np.mean(h_zero_ablate))
    h_orig = float(np.mean(h_original))
    loss_recovered, loss_recovered_valid = compute_loss_recovered(h_with, h_zero, h_orig)
    mse_reduction_vs_zero = (
        (h_zero - h_with) / h_zero if h_zero > 1e-8 else 0.0
    )

    out: Dict[str, float] = {}
    if h_zero_cls_proxy:
        h_zero_cls = float(np.mean(h_zero_cls_proxy))
        lr_cls, lr_cls_valid = compute_loss_recovered(h_with, h_zero_cls, h_orig)
        out.update(
            {
                "H_zero_cls_proxy": h_zero_cls,
                "loss_recovered_cls_proxy": float(lr_cls),
                "loss_recovered_cls_proxy_valid": lr_cls_valid,
                "mse_reduction_vs_cls_proxy": (
                    (h_zero_cls - h_with) / h_zero_cls if h_zero_cls > 1e-8 else 0.0
                ),
            }
        )

    fired = torch.zeros(n_features, dtype=torch.bool)
    for start in tqdm(batch_starts, desc="Eval dead latents", unit="batch", leave=False):
        batch = acts[start : start + batch_size]
        encoded_acts, _ = sae.encode(batch)
        fired |= (encoded_acts > 0).any(dim=0).cpu()
    dead_fraction = float(1.0 - fired.float().mean().item())

    out.update(
        {
            "loss_recovered": float(loss_recovered),
            "loss_recovered_valid": loss_recovered_valid,
            "mse_reduction_vs_zero": float(mse_reduction_vs_zero),
            "mean_L0": float(np.mean(l0_vals)),
            "dead_fraction": dead_fraction,
            "H_with_sae": h_with,
            "H_zero_ablate": h_zero,
            "H_original": h_orig,
        }
    )
    return out


def compute_loss_recovered(
    h_with: float,
    h_zero: float,
    h_orig: float,
    denom_eps: float = 1e-8,
) -> Tuple[float, bool]:
    """
    Fraction of the (zero-ablate − dense) reconstruction gap closed by the SAE.

    When the dense encoder baseline is not better than zero ablation (common at
    early ViT layers), the ratio is undefined — fall back to mse_reduction_vs_zero
    and mark loss_recovered_valid=False.
    """
    gap_sae = h_zero - h_with
    gap_dense = h_zero - h_orig
    if gap_sae <= denom_eps:
        return 0.0, gap_dense > denom_eps
    if gap_dense <= denom_eps:
        return gap_sae / max(h_zero, denom_eps), False
    return gap_sae / gap_dense, True


@torch.no_grad()
def compute_feature_activation_matrix(
    sae: nn.Module,
    backbone: BackboneAdapter,
    eval_dataset: Subset,
    eval_loader: DataLoader,
    layer_index: int,
    norm_spec: dict,
    device: str,
) -> np.ndarray:
    """
    Build [n_features, n_images] matrix of max latent activation per image.
    Uses mean patch-token activation per image for each latent.
    """
    n_features = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features
    n_images = len(eval_dataset)
    matrix = np.zeros((n_features, n_images), dtype=np.float32)

    backbone.model.eval()
    sae.eval()

    image_offset = 0
    for batch in tqdm(eval_loader, desc="Feature activation matrix", unit="batch", leave=False):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            indices = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            indices = None

        images = images.to(device, non_blocking=True)
        b = images.shape[0]

        hook_storage = {}

        def make_hook():
            def hook(module, inputs, output):
                patch_tokens = output[:, backbone.spec.num_prefix_tokens :, :]
                hook_storage["tokens"] = patch_tokens

            return hook

        handle = backbone.blocks[layer_index].register_forward_hook(make_hook())
        backbone.forward(images)
        handle.remove()

        tokens = hook_storage["tokens"]
        flat = tokens.reshape(-1, tokens.shape[-1])
        flat = preprocess_activations(flat, norm_spec).to(device)
        encoded, _ = sae.encode(flat)
        patches_per_image = tokens.shape[1]
        encoded = encoded.reshape(b, patches_per_image, -1)
        max_per_image = encoded.max(dim=1).values.cpu().numpy()

        if indices is None:
            idxs = np.arange(image_offset, image_offset + b)
            image_offset += b
        else:
            idxs = indices.numpy() if torch.is_tensor(indices) else np.asarray(indices)

        matrix[:, idxs] = max_per_image.T

    return matrix


@torch.no_grad()
def absorption_proxy(
    feature_matrix: np.ndarray,
    top_k: int = 20,
    shared_threshold: float = 0.5,
) -> float:
    """
    For each image's top-k latents, count fraction that fire on >50% of images.
    Higher values indicate more feature absorption / polysemanticity.
    """
    n_features, n_images = feature_matrix.shape
    if n_images == 0:
        return 0.0

    shared_count = 0
    total_count = 0

    image_active = feature_matrix > 0
    latent_image_fraction = image_active.sum(axis=1) / n_images

    for img_idx in range(n_images):
        top_latents = np.argsort(-feature_matrix[:, img_idx])[:top_k]
        for latent in top_latents:
            total_count += 1
            if latent_image_fraction[latent] > shared_threshold:
                shared_count += 1

    return float(shared_count / total_count) if total_count else 0.0
