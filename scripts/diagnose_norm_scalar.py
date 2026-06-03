#!/usr/bin/env python3
"""Compare SAE eval metrics with inference norm (1.0) vs train norm_scalar."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import evaluate_sae_on_activations
from src.sae.trainer import load_sae_checkpoint


def load_cached_acts(run_dir: Path) -> torch.Tensor:
    path = run_dir / "eval_patch_activations.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, dict):
        for key in ("activations", "acts", "patch_activations"):
            if key in data:
                return data[key]
        return next(v for v in data.values() if isinstance(v, torch.Tensor))
    raise TypeError(f"Unexpected cache type: {type(data)}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [
        ("croma", "early", "matryoshka"),
        ("prithvi_eo_2", "late", "topk"),
        ("mars_orbital_vit", "late", "topk"),
        ("momo", "late", "topk"),
    ]
    for backbone, layer, sae_arch in runs:
        run_dir = PROJECT_ROOT / "results" / backbone / layer / sae_arch
        acts = load_cached_acts(run_dir)
        sae, payload = load_sae_checkpoint(run_dir / "sae_weights.pt", device)
        ns = float(payload["norm_scalar"])
        folded = payload.get("norm_folded")
        print(f"\n=== {backbone}/{layer}/{sae_arch} ===")
        print(f"  norm_scalar={ns:.4f} norm_folded={folded}")
        for scalar, label in [(1.0, "inference (1.0)"), (ns, "train_norm")]:
            m = evaluate_sae_on_activations(
                sae, acts, norm_scalar=scalar, device=device, batch_size=2048
            )
            print(
                f"  {label}: mse_red={m['mse_reduction_vs_zero']:.4f} "
                f"H_with={m['H_with_sae']:.4f} valid={m['loss_recovered_valid']}"
            )


def test_double_fold(backbone: str, layer: str, sae_arch: str) -> None:
    """If extra fold improves metrics, checkpoint was saved without fold despite flag."""
    import copy

    from src.sae.utils import fold_norm_scalar_into_sae

    run_dir = PROJECT_ROOT / "results" / backbone / layer / sae_arch
    acts = load_cached_acts(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, payload = load_sae_checkpoint(run_dir / "sae_weights.pt", device)
    ns = float(payload["norm_scalar"])
    m0 = evaluate_sae_on_activations(sae, acts, 1.0, device, 2048)["mse_reduction_vs_zero"]
    sae2 = copy.deepcopy(sae)
    fold_norm_scalar_into_sae(sae2, ns)
    m1 = evaluate_sae_on_activations(sae2, acts, 1.0, device, 2048)["mse_reduction_vs_zero"]
    print(f"\n[double-fold test] {backbone}/{layer}/{sae_arch}")
    print(f"  as loaded @1.0:     mse_red={m0:.4f}")
    print(f"  extra fold @1.0:    mse_red={m1:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--double-fold-test", action="store_true")
    args, _ = parser.parse_known_args()
    if args.double_fold_test:
        for b, l, s in [
            ("croma", "early", "matryoshka"),
            ("prithvi_eo_2", "late", "topk"),
            ("mars_orbital_vit", "late", "topk"),
        ]:
            test_double_fold(b, l, s)
    else:
        main()
