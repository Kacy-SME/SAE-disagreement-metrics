"""
Observation Diversity Score (obs_div) for post-2025 HiRISE SAE latents.

For each latent, fraction of top-k activating patches from distinct observation IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.base import BackboneAdapter
from src.sae.activation_norm import preprocess_activations


@torch.no_grad()
def compute_observation_diversity(
    sae: nn.Module,
    backbone: BackboneAdapter,
    dataloader: DataLoader,
    obs_ids: Sequence[str],
    layer_index: int,
    norm_spec: dict,
    device: str,
    top_k: int = 20,
) -> Dict[str, float | np.ndarray]:
    """
    Per-latent obs_div in [0, 1]: distinct obs_ids among global top-k patch activations / top_k.
    """
    n_features = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features
    n_patches = len(obs_ids)
    if n_patches == 0:
        return {
            "obs_div_mean": float("nan"),
            "obs_div_median": float("nan"),
            "obs_div_topk": top_k,
            "obs_div_per_latent": np.array([], dtype=np.float32),
        }

    max_acts = np.zeros((n_features, n_patches), dtype=np.float32)
    patch_offset = 0

    backbone.model.eval()
    sae.eval()

    for batch in tqdm(dataloader, desc="obs_div activations", unit="batch", leave=False):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            indices = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            indices = None

        images = images.to(device, non_blocking=True)
        b = images.shape[0]
        storage: dict = {}

        def hook_fn(module, inputs, output):
            patch_tokens = output[:, backbone.spec.num_prefix_tokens :, :]
            storage["tokens"] = patch_tokens

        handle = backbone.blocks[layer_index].register_forward_hook(hook_fn)
        backbone.forward(images)
        handle.remove()

        tokens = storage["tokens"]
        flat = tokens.reshape(-1, tokens.shape[-1])
        flat = preprocess_activations(flat, norm_spec).to(device)
        encoded, _ = sae.encode(flat)
        patches_per_image = tokens.shape[1]
        encoded = encoded.reshape(b, patches_per_image, -1)
        max_per_patch = encoded.max(dim=1).values.cpu().numpy()

        if indices is None:
            idxs = np.arange(patch_offset, patch_offset + b)
            patch_offset += b
        else:
            idxs = indices.numpy() if torch.is_tensor(indices) else np.asarray(indices)

        max_acts[:, idxs] = max_per_patch.T

    per_latent = np.zeros(n_features, dtype=np.float32)
    obs_arr = np.asarray(list(obs_ids))
    k = min(top_k, n_patches)

    for latent in range(n_features):
        top_patch_idxs = np.argsort(-max_acts[latent])[:k]
        distinct = len(set(obs_arr[top_patch_idxs].tolist()))
        per_latent[latent] = distinct / float(k)

    alive = (max_acts > 0).any(axis=1)
    alive_scores = per_latent[alive] if alive.any() else per_latent

    return {
        "obs_div_mean": float(np.mean(alive_scores)),
        "obs_div_median": float(np.median(alive_scores)),
        "obs_div_fraction_high": float((alive_scores >= 0.5).mean()) if alive_scores.size else 0.0,
        "obs_div_topk": k,
        "obs_div_per_latent": per_latent,
        "obs_div_n_alive": int(alive.sum()),
    }


def save_obs_div_per_latent_csv(per_latent: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"latent_idx": np.arange(len(per_latent)), "obs_div": per_latent})
    df.to_csv(out_path, index=False)


def obs_div_summary_row(metrics: Dict) -> Dict[str, float]:
    return {
        "obs_div_mean": metrics.get("obs_div_mean", float("nan")),
        "obs_div_median": metrics.get("obs_div_median", float("nan")),
        "obs_div_fraction_high": metrics.get("obs_div_fraction_high", float("nan")),
        "obs_div_topk": metrics.get("obs_div_topk", 20),
        "obs_div_n_alive": metrics.get("obs_div_n_alive", 0),
    }
