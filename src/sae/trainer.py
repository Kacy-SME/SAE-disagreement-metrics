"""SAE training loop with activation buffer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.base import BackboneAdapter
from src.extract.activations import (
    ActivationBuffer,
    stream_patch_activations_to_buffer,
)
from src.sae.matryoshka import MatryoshkaBatchTopKSAE
from src.sae.topk import TopKSAE
from src.sae.activation_norm import preprocess_activations
from src.sae.utils import (
    fold_norm_scalar_into_sae,
)


def build_sae(
    arch: str,
    hidden_dim: int,
    sae_cfg: Dict[str, Any],
    device: str,
) -> nn.Module:
    k = int(sae_cfg.get("k", 40))
    k_aux = hidden_dim // int(sae_cfg.get("k_aux_divisor", 2))
    if arch == "topk":
        dict_size = hidden_dim * int(sae_cfg.get("dictionary_multiplier", 4))
        sae = TopKSAE(d_in=hidden_dim, dict_size=dict_size, k=k, k_aux=k_aux)
    elif arch == "matryoshka":
        multipliers = sae_cfg.get("nested_multipliers", [1, 2, 4])
        nested_sizes = [hidden_dim * m for m in multipliers]
        sae = MatryoshkaBatchTopKSAE(
            d_in=hidden_dim,
            nested_sizes=nested_sizes,
            k=k,
            k_aux=k_aux,
        )
    else:
        raise ValueError(f"Unknown SAE arch {arch!r}")
    return sae.to(device)


def refill_buffer(
    buffer: ActivationBuffer,
    backbone: BackboneAdapter,
    dataloader: DataLoader,
    layer_index: int,
    device: str,
) -> None:
    stream_patch_activations_to_buffer(
        backbone=backbone,
        dataloader=dataloader,
        layer_index=layer_index,
        buffer=buffer,
        device=device,
        desc="Refill activation buffer",
    )


def train_sae(
    sae: nn.Module,
    buffer: ActivationBuffer,
    backbone: BackboneAdapter,
    train_loader: DataLoader,
    layer_index: int,
    cfg: Dict[str, Any],
    device: str,
    norm_spec: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    steps = int(cfg["steps"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    log_every = int(cfg.get("log_every", 100))
    aux_weight = float(cfg.get("aux_loss_weight", 1.0))

    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)

    recon_curve: List[float] = []
    sparsity_curve: List[float] = []

    pbar = tqdm(range(steps), desc="SAE training", unit="step", leave=False)
    for step in pbar:
        if buffer.needs_refill(training_step=step):
            pbar.set_postfix_str("refilling buffer from ViT...")
            refill_buffer(buffer, backbone, train_loader, layer_index, device)
            buffer.mark_refilled(training_step=step)
            pbar.set_postfix_str("")

        batch = buffer.sample(batch_size).to(device)
        batch = preprocess_activations(batch, norm_spec)

        optimizer.zero_grad(set_to_none=True)
        loss_out = sae.loss(batch, aux_weight=aux_weight)
        loss_out.total.backward()
        optimizer.step()
        sae.post_grad_step()

        with torch.no_grad():
            acts, _ = sae.encode(batch)
            sae.update_dead_latent_stats(acts)

        recon_curve.append(float(loss_out.recon.detach().cpu()))
        sparsity_curve.append(float(loss_out.aux.detach().cpu()))

        if step % log_every == 0:
            pbar.set_postfix(
                recon=f"{loss_out.recon.item():.4f}",
                aux=f"{loss_out.aux.item():.4f}",
                l0=f"{loss_out.l0.item():.1f}",
                refresh=False,
            )

    pbar.close()

    curve = np.stack([recon_curve, sparsity_curve], axis=1)
    stats = compute_activation_stats(
        sae, buffer, device, norm_spec, sample_steps=20
    )
    return curve, stats


@torch.no_grad()
def compute_activation_stats(
    sae: nn.Module,
    buffer: ActivationBuffer,
    device: str,
    norm_spec: Dict[str, Any],
    sample_steps: int = 20,
    batch_size: int = 2048,
) -> Dict[str, float]:
    l0_vals = []
    norms = []
    n_latents = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features
    dead = torch.zeros(n_latents, dtype=torch.bool)

    for _ in range(sample_steps):
        if len(buffer) < batch_size:
            break
        batch = buffer.sample(batch_size).to(device)
        batch = preprocess_activations(batch, norm_spec)
        if hasattr(sae, "encode"):
            acts, _ = sae.encode(batch)
        else:
            acts = sae(batch)[1]
        l0_vals.append((acts > 0).float().sum(dim=-1).mean().item())
        norms.append(acts.norm(dim=-1).mean().item())
        dead |= (acts > 0).any(dim=0).cpu()

    return {
        "mean_L0": float(np.mean(l0_vals)) if l0_vals else 0.0,
        "dead_fraction": float(1.0 - dead.float().mean().item()),
        "mean_activation_norm": float(np.mean(norms)) if norms else 0.0,
    }


def save_sae_checkpoint(
    path: Path,
    sae: nn.Module,
    norm_spec: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    spec = dict(norm_spec)
    if spec.get("preprocess_mode", "scalar") == "scalar":
        fold_norm_scalar_into_sae(sae, float(spec.get("norm_scalar", 1.0)))
        spec["norm_scalar"] = float(spec.get("norm_scalar", 1.0))
    else:
        spec["norm_scalar"] = 1.0
    payload = {
        "state_dict": sae.state_dict(),
        "config": sae.config_dict(),
        "norm_scalar": spec["norm_scalar"],
        "norm_folded": True,
        "preprocess_mode": spec.get("preprocess_mode", "scalar"),
        "center_activations": bool(spec.get("center_activations", False)),
        "meta": meta,
    }
    if spec.get("act_mean") is not None:
        payload["act_mean"] = spec["act_mean"].cpu()
    if spec.get("act_std") is not None:
        payload["act_std"] = spec["act_std"].cpu()
    if spec.get("pca_components") is not None:
        payload["pca_components"] = spec["pca_components"].cpu()
        payload["pca_k_95"] = int(spec.get("pca_k_95", spec["pca_components"].shape[0]))
        payload["projected_dim"] = int(
            spec.get("projected_dim", spec["pca_components"].shape[0])
        )
        if spec.get("pca_variance_threshold") is not None:
            payload["pca_variance_threshold"] = float(spec["pca_variance_threshold"])
    if spec.get("zca_matrix") is not None:
        payload["zca_matrix"] = spec["zca_matrix"].cpu()
        payload["zca_eps"] = float(spec.get("zca_eps", 1e-3))
    torch.save(payload, path)


@torch.no_grad()
def verify_checkpoint_norm_consistency(
    weights_path: Path,
    backbone: str,
    layer_index: int,
    eval_acts: torch.Tensor,
    device: str,
    tol: float = 0.15,
) -> None:
    """
    Sanity-check that checkpoint norm folding matches eval activations.

    Uses the first 512 patch vectors from eval_acts and the inference norm scalar
    from the checkpoint payload. Raises if reconstruction vs zero-ablation MSE
    reduction is too low (typical sign of norm_folded=True on unfolded weights).
    """
    min_mse_reduction = 0.3
    from src.sae.activation_norm import norm_spec_for_eval, preprocess_activations

    sae, payload = load_sae_checkpoint(weights_path, device)
    norm_spec = norm_spec_for_eval(payload)

    n = min(512, eval_acts.shape[0])
    if n == 0:
        raise RuntimeError(
            f"Checkpoint norm consistency check failed: no eval activations for "
            f"{backbone} layer {layer_index} ({weights_path})."
        )

    acts = preprocess_activations(eval_acts[:n].float(), norm_spec).to(device)
    encoded, _ = sae.encode(acts)
    if hasattr(sae, "nested_reconstructions"):
        recon = sae.nested_reconstructions(acts, encoded)[-1]
    else:
        recon = sae.decode(encoded)
    zero_recon = sae.decode(torch.zeros_like(encoded))

    mse_recon = float(F.mse_loss(recon, acts).item())
    mse_zero = float(F.mse_loss(zero_recon, acts).item())
    if mse_zero <= 1e-8:
        mse_reduction = 0.0
    else:
        mse_reduction = 1.0 - mse_recon / mse_zero

    if mse_reduction < min_mse_reduction:
        raise RuntimeError(
            f"Checkpoint norm consistency check failed (mse_reduction={mse_reduction:.3f}). "
            f"Checkpoint may have been saved with incorrect norm_folded flag. "
            f"Retrain with --force. "
            f"(backbone={backbone}, layer_index={layer_index}, path={weights_path}, "
            f"norm_folded={payload.get('norm_folded')}, stored_norm_scalar={payload.get('norm_scalar')}, "
            f"preprocess_mode={payload.get('preprocess_mode', 'scalar')}, tol={tol})"
        )


def load_sae_checkpoint(path: Path, device: str) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["arch"] == "topk":
        sae = TopKSAE(
            d_in=cfg["d_in"],
            dict_size=cfg["dict_size"],
            k=cfg["k"],
            k_aux=cfg["k_aux"],
        )
    else:
        sae = MatryoshkaBatchTopKSAE(
            d_in=cfg["d_in"],
            nested_sizes=cfg["nested_sizes"],
            k=cfg["k"],
            k_aux=cfg["k_aux"],
        )
    sae.load_state_dict(payload["state_dict"])
    sae.to(device)
    sae.eval()
    return sae, payload
