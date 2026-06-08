"""CROMA optical encoder backbone (s2_encoder from official checkpoint)."""

from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from src.backbones.base import BackboneAdapter, BackboneSpec


def _patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = imgs.shape
    nh, nw = h // patch_size, w // patch_size
    x = imgs.reshape(b, c, nh, patch_size, nw, patch_size)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, nh * nw, c * patch_size * patch_size)


def get_2dalibi(num_heads: int, num_patches: int) -> torch.Tensor:
    side = int(math.sqrt(num_patches))
    points = list(itertools.product(range(side), range(side)))

    def get_slopes(n: int) -> list[float]:
        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest = 2 ** math.floor(math.log2(n))
        extra = get_slopes(2 * closest)[0::2][: n - closest]
        return get_slopes_power_of_2(closest) + extra

    slopes = torch.tensor(get_slopes(num_heads), dtype=torch.float32).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class CROMAFFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.input_norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(x))


class CROMAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, relative_position_bias: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        b, n, _ = q.shape
        d = q.shape[-1] // self.num_heads
        q = q.view(b, n, self.num_heads, d).transpose(1, 2)
        k = k.view(b, n, self.num_heads, d).transpose(1, 2)
        v = v.view(b, n, self.num_heads, d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + relative_position_bias.to(scores.device, dtype=scores.dtype)
        attn = self.dropout(scores.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class CROMAEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.attn = CROMAAttention(dim=dim, num_heads=num_heads)
        self.ffn = CROMAFFN(dim=dim)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, attn_bias) + x
        x = self.ffn(x) + x
        return x


class CROMAOpticalViT(nn.Module):
    """Official CROMA s2_encoder (12-channel optical ViT, patch size 8)."""

    PATCH_SIZE = 8
    IN_CHANNELS = 12
    NUM_HEADS = 16

    def __init__(
        self,
        image_size: int = 128,
        dim: int = 768,
        depth: int = 12,
    ):
        super().__init__()
        self.image_size = image_size
        self.dim = dim
        self.depth = depth
        self.num_patches = (image_size // self.PATCH_SIZE) ** 2
        pixels_per_patch = self.PATCH_SIZE * self.PATCH_SIZE * self.IN_CHANNELS
        self.linear_input = nn.Linear(pixels_per_patch, dim)
        self.layers = nn.ModuleList(
            [CROMAEncoderLayer(dim=dim, num_heads=self.NUM_HEADS) for _ in range(depth)]
        )
        self.register_buffer(
            "attn_bias",
            get_2dalibi(self.NUM_HEADS, self.num_patches),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _patchify(x, self.PATCH_SIZE)
        x = self.linear_input(x)
        bias = self.attn_bias
        for layer in self.layers:
            x = layer(x, bias)
        return x


class CROMAAdapter(BackboneAdapter):
    def __init__(
        self,
        spec: BackboneSpec,
        device: str = "cuda",
        hf_repo: str = "antofuller/CROMA",
        hf_filename: str = "CROMA_base.pt",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(spec, device)
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.checkpoint_path = checkpoint_path
        self.model: nn.Module = nn.Identity()

    def _resolve_checkpoint(self) -> str:
        if self.checkpoint_path:
            return self.checkpoint_path
        return hf_hub_download(repo_id=self.hf_repo, filename=self.hf_filename)

    @staticmethod
    def _rgb_to_12ch(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 12:
            return x
        if x.shape[1] != 3:
            raise ValueError(f"CROMA expects 3- or 12-channel input, got {x.shape[1]}")
        return x.repeat(1, 4, 1, 1)

    def load(self) -> None:
        image_size = self.spec.image_size
        if image_size % 8 != 0:
            raise ValueError(f"CROMA image_size must be divisible by 8, got {image_size}")

        self.model = CROMAOpticalViT(
            image_size=image_size,
            dim=self.spec.hidden_dim,
            depth=self.spec.num_layers,
        )
        ckpt_path = self._resolve_checkpoint()
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict) or "s2_encoder" not in raw:
            raise KeyError(f"CROMA checkpoint missing s2_encoder: {ckpt_path}")
        state = raw["s2_encoder"]
        remapped: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            key = k.replace("module.", "")
            if key.startswith("transformer.layers."):
                rest = key[len("transformer.layers.") :]
                layer_idx, sub_idx, *tail = rest.split(".")
                sub = "attn" if sub_idx == "0" else "ffn"
                suffix = ".".join(tail) if tail else ""
                key = f"layers.{layer_idx}.{sub}" + (f".{suffix}" if suffix else "")
            remapped[key] = v

        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        print(
            f"[CROMA] checkpoint={ckpt_path} missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[CROMA] sample missing: {missing[:5]}")
        self.model.eval()

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.layers

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> None:
        self.model(self._rgb_to_12ch(x))
