"""
AnySat backbone adapter (stub).

TODO: Full AnySat integration via torch.hub.load('gastruc/anysat', 'anysat').
AnySat expects a modality dict (e.g. spot RGB at 1m) and scale-adaptive JEPA
forward — not standard timm ViT blocks. For HiRISE RED, replicate the band
3× and use the spot modality with normalized (x-mean)/std inputs, then call
AnySat(data, patch_size=..., output='tile'|'patch'|'all').

Until integrated, this stub exposes ViT-shaped blocks with a fixed random
projection so the SAE training / hook pipeline can run end-to-end.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.backbones.base import BackboneAdapter, BackboneSpec

# AnySat-base tile embedding dimension (placeholder; update when hub model wired).
ANYSAT_HIDDEN_DIM = 768
ANYSAT_NUM_LAYERS = 12


class _StubBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.norm(x))


class _AnySatStubViT(nn.Module):
    """Minimal ViT-shaped module for activation hooks."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = ANYSAT_HIDDEN_DIM,
        num_layers: int = ANYSAT_NUM_LAYERS,
        seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        grid = image_size // patch_size
        self.num_patches = grid * grid

        torch.manual_seed(seed)
        self.patch_embed = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        nn.init.orthogonal_(self.patch_embed.weight.flatten(1))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [_StubBlock(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class AnySatAdapter(BackboneAdapter):
    def __init__(
        self,
        spec: BackboneSpec,
        device: str = "cuda",
        stub_seed: int = 42,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(spec, device)
        self.stub_seed = stub_seed
        self.checkpoint_path = checkpoint_path
        self.model: nn.Module = nn.Identity()

    def load(self) -> None:
        if self.checkpoint_path:
            print(
                f"[AnySat] TODO: load checkpoint {self.checkpoint_path}; "
                "using random-projection stub."
            )
        else:
            print(
                "[AnySat] TODO: wire torch.hub AnySat forward for HiRISE RED->RGB; "
                "using random-projection stub."
            )
        self.model = _AnySatStubViT(
            image_size=self.spec.image_size,
            patch_size=self.spec.patch_size,
            hidden_dim=self.spec.hidden_dim,
            num_layers=self.spec.num_layers,
            seed=self.stub_seed,
        )
        self.model.eval()

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.blocks
