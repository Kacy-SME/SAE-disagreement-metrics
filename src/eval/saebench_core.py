"""
SAEBench Core eval metrics (ported from sae_bench/evals/core).

Reference: https://github.com/adamkarvonen/SAEBench
Does not require transformer_lens — operates on activation tensors + SAE encode/decode.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.sae.activation_norm import norm_spec_for_eval, preprocess_activations


@torch.no_grad()
def compute_core_metrics(
    sae: nn.Module,
    activations: torch.Tensor,
    norm_spec: dict,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """
    SAEBench-style reconstruction / sparsity / variance metrics on flat activations [N, d_in].
    """
    device = next(sae.parameters()).device
    # Keep activations on CPU; move batches to GPU only (avoids OOM on 100k+ patches).
    acts_cpu = preprocess_activations(activations.float(), norm_spec)
    n = acts_cpu.shape[0]
    dict_size = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features

    l0_list = []
    l1_list = []
    mse_list = []
    cossim_list = []
    l2_ratio_list = []
    mean_sum_of_squares = []
    mean_act_per_dimension = []
    mean_sum_of_resid_squared = []
    explained_variance_legacy_list = []

    for start in range(0, n, batch_size):
        batch = acts_cpu[start : start + batch_size].to(device)
        if hasattr(sae, "encode"):
            feat_acts, _ = sae.encode(batch)
            if hasattr(sae, "nested_reconstructions"):
                recons = sae.nested_reconstructions(batch, feat_acts)
                recon = recons[-1]
            else:
                recon = sae.decode(feat_acts)
        else:
            recon, feat_acts, _ = sae(batch)

        resid = batch - recon
        l0_list.append((feat_acts != 0).sum(dim=-1).float())
        l1_list.append(feat_acts.sum(dim=-1))
        mse_list.append(resid.pow(2).mean(dim=-1))
        x_norm = batch / batch.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_hat_norm = recon / recon.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cossim_list.append((x_norm * x_hat_norm).sum(dim=-1))

        l2_in = batch.norm(dim=-1)
        l2_out = recon.norm(dim=-1)
        l2_ratio_list.append(l2_out / l2_in.clamp_min(1e-4))

        resid_ss = resid.pow(2).sum(dim=-1)
        batched_var = (batch - batch.mean(dim=0)).pow(2).sum(dim=-1).clamp_min(1e-8)
        explained_variance_legacy_list.append(1 - resid_ss / batched_var)

        mean_sum_of_squares.append(batch.pow(2).sum(dim=-1).mean(dim=0))
        mean_act_per_dimension.append(batch.pow(2).mean(dim=0))
        mean_sum_of_resid_squared.append(resid_ss.mean(dim=0))

    l0 = torch.cat(l0_list)
    l1 = torch.cat(l1_list)
    mse = torch.cat(mse_list)
    cossim = torch.cat(cossim_list)

    mss = torch.stack(mean_sum_of_squares).mean(dim=0)
    mad = torch.cat(mean_act_per_dimension).mean(dim=0)
    total_variance = mss - mad**2
    residual_variance = torch.stack(mean_sum_of_resid_squared).mean(dim=0)
    explained_variance = (1 - residual_variance / total_variance.clamp_min(1e-8)).item()

    h_with = mse.mean().item()
    h_zero_parts = []
    for start in range(0, n, batch_size):
        batch = acts_cpu[start : start + batch_size].to(device)
        z = torch.zeros(batch.shape[0], dict_size, device=device, dtype=batch.dtype)
        h_zero_parts.append((batch - sae.decode(z)).pow(2).mean(dim=-1).cpu())
    h_zero = torch.cat(h_zero_parts).mean().item()

    loss_recovered = (h_with - h_zero) / max(1e-8, h_with) if h_with > 0 else 0.0
    # SAEBench core also reports CE/KL on LM; we report activation MSE analogue:
    mse_loss_recovered = 1.0 - h_with / h_zero if h_zero > 1e-8 else 0.0

    return {
        "core_mean_l0": float(l0.mean().item()),
        "core_mean_l1": float(l1.mean().item()),
        "core_mse": float(h_with),
        "core_mse_loss_recovered_vs_zero": float(max(0.0, min(1.0, mse_loss_recovered))),
        "core_explained_variance": float(explained_variance),
        "core_explained_variance_legacy": float(
            torch.cat(explained_variance_legacy_list).mean().item()
        ),
        "core_cossim": float(cossim.mean().item()),
        "core_l2_ratio": float(torch.cat(l2_ratio_list).mean().item()),
    }


def load_sae_for_eval(
    weights_path, device: str
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    from src.sae.trainer import load_sae_checkpoint

    sae, payload = load_sae_checkpoint(weights_path, device)
    norm_spec = norm_spec_for_eval(payload)
    return sae, payload, norm_spec
