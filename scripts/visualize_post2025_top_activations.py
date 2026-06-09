#!/usr/bin/env python3
"""Top-activating post-2025 patches with obs_id — saved per backbone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones.registry import create_backbone, load_backbone_configs  # noqa: E402
from src.data.post_may2025 import (  # noqa: E402
    PostMay2025IndexedDataset,
    _load_patch_array,
    _scale_to_uint8,
    build_post_may2025_datasets,
    default_patch_cache_dir,
)
from src.eval.saebench_core import load_sae_for_eval  # noqa: E402
from src.extract.activations import activation_dataloader  # noqa: E402
from src.sae.activation_norm import preprocess_activations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--layer-depth", default="middle")
    parser.add_argument("--sae-arch", default="topk")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--top-k-latents", type=int, default=5)
    parser.add_argument("--post2025-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "post2025_top_activations",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    weights = args.run_dir / "sae_weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    backbone_cfgs = load_backbone_configs()
    combo = None
    from src.backbones.registry import list_experiment_combinations

    for c in list_experiment_combinations(
        backbone_names=[args.backbone],
        layer_depths=[args.layer_depth],
        sae_archs=[args.sae_arch],
        configs=backbone_cfgs,
    ):
        combo = c
        break
    if combo is None:
        raise ValueError(f"No combo for {args.backbone}/{args.layer_depth}/{args.sae_arch}")

    sae, _, norm_spec = load_sae_for_eval(weights, device)
    backbone = create_backbone(args.backbone, device=device, configs=backbone_cfgs)
    transform = backbone.get_transform()
    _train, _eval, _info, _te, eval_entries = build_post_may2025_datasets(
        cache_dir=args.post2025_cache_dir or default_patch_cache_dir(),
        transform=transform,
    )
    n_eval = min(500, len(eval_entries))
    eval_slice = eval_entries[:n_eval]
    eval_ds = PostMay2025IndexedDataset(eval_slice, transform=transform)
    loader = activation_dataloader(eval_ds, batch_size=32, num_workers=0)

    layer_index = int(combo["layer_index"])
    max_acts = np.zeros((sae.dict_size, n_eval), dtype=np.float32)
    patch_offset = 0

    for batch in loader:
        images, indices = batch[0], batch[1]
        images = images.to(device)
        storage: dict = {}

        def hook_fn(module, inputs, output):
            storage["tokens"] = output[:, backbone.spec.num_prefix_tokens :, :]

        handle = backbone.blocks[layer_index].register_forward_hook(hook_fn)
        backbone.forward(images)
        handle.remove()
        tokens = storage["tokens"]
        flat = tokens.reshape(-1, tokens.shape[-1])
        flat = preprocess_activations(flat, norm_spec).to(device)
        encoded, _ = sae.encode(flat)
        b = tokens.shape[0]
        encoded = encoded.reshape(b, tokens.shape[1], -1).max(dim=1).values.cpu().numpy()
        idxs = indices.numpy() if torch.is_tensor(indices) else np.asarray(indices)
        max_acts[:, idxs] = encoded.T
        patch_offset += b

    out_dir = args.output_dir / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    alive = (max_acts > 0).any(axis=1)
    latent_order = np.argsort(-max_acts.max(axis=1))
    latent_order = [i for i in latent_order if alive[i]][: args.top_k_latents]

    for latent in latent_order:
        top_patch_idxs = np.argsort(-max_acts[latent])[: args.top_n]
        fig, axes = plt.subplots(2, (args.top_n + 1) // 2, figsize=(12, 5))
        axes = np.atleast_1d(axes).flatten()
        for ax_i, patch_idx in enumerate(top_patch_idxs):
            entry = eval_slice[int(patch_idx)]
            patch = _load_patch_array(Path(entry["patch_path"]))
            gray = _scale_to_uint8(np.asarray(patch, dtype=np.float32))
            obs_id = entry["obs_id"]
            ax = axes[ax_i]
            ax.imshow(gray, cmap="gray")
            ax.set_title(f"{obs_id}\nact={max_acts[latent, patch_idx]:.3f}", fontsize=7)
            ax.axis("off")
        for ax in axes[len(top_patch_idxs) :]:
            ax.axis("off")
        fig.suptitle(f"Latent {latent} — top {args.top_n} post-2025 patches")
        fig.tight_layout()
        fig.savefig(out_dir / f"latent_{latent:04d}_top_patches.png", dpi=150)
        plt.close(fig)

    print(f"Wrote top-activation figures -> {out_dir}")


if __name__ == "__main__":
    main()
