"""Custom Mars Orbital DINOv2-style ViT with register tokens."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed

from src.backbones.base import BackboneAdapter, BackboneSpec


class MarsOrbitalViT(nn.Module):
    """Minimal Mars Orbital ViT backbone matching Orbital_ViT pretraining."""

    def __init__(
        self,
        timm_model: str = "vit_base_patch14_dinov2",
        embed_dim: int = 768,
        num_register_tokens: int = 4,
        image_size: int = 224,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.backbone = timm.create_model(
            timm_model,
            pretrained=False,
            num_classes=0,
            img_size=image_size,
            dynamic_img_size=True,
        )
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def _tokens_with_pos_and_registers(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        patches = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            if patches.ndim == 4 and self.backbone.dynamic_img_size:
                _, h, w, _ = patches.shape
                pos_embed = resample_abs_pos_embed(
                    self.backbone.pos_embed,
                    new_size=(h, w),
                    old_size=self.backbone.patch_embed.grid_size,
                    num_prefix_tokens=self.backbone.num_prefix_tokens,
                )
                patches = patches.reshape(b, h * w, -1)
            else:
                pos_embed = self.backbone.pos_embed
                if patches.ndim == 4:
                    patches = patches.reshape(b, -1, patches.shape[-1])
            pos_cls = pos_embed[:, :1, :]
            pos_patch = pos_embed[:, 1:, :]
            patches = patches + pos_patch
            cls_token = self.backbone.cls_token.expand(b, -1, -1) + pos_cls
        else:
            if patches.ndim == 4:
                patches = patches.reshape(b, -1, patches.shape[-1])
            cls_token = self.backbone.cls_token.expand(b, -1, -1)
        reg_tokens = self.register_tokens.expand(b, -1, -1)
        return self.backbone.pos_drop(torch.cat([cls_token, reg_tokens, patches], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens_with_pos_and_registers(x)
        for blk in self.backbone.blocks:
            tokens = blk(tokens)
        return self.backbone.norm(tokens)


class MarsOrbitalViTAdapter(BackboneAdapter):
    def __init__(
        self,
        spec: BackboneSpec,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        num_register_tokens: int = 4,
        timm_model: str = "vit_base_patch14_dinov2",
    ):
        super().__init__(spec, device)
        self.checkpoint_path = checkpoint_path
        self.num_register_tokens = num_register_tokens
        self.timm_model = timm_model
        self.model = nn.Identity()

    def load(self) -> None:
        if not self.checkpoint_path:
            raise FileNotFoundError(
                "Mars Orbital ViT checkpoint_path is required "
                "(set in configs/backbones.yaml or MARS_DRIVE_ROOT)."
            )
        self.model = MarsOrbitalViT(
            timm_model=self.timm_model,
            embed_dim=self.spec.hidden_dim,
            num_register_tokens=self.num_register_tokens,
            image_size=self.spec.image_size,
        )
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model.backbone.load_state_dict(state, strict=True)
        self.model.eval()
        from tqdm import tqdm

        tqdm.write(f"[Mars_Orbital_ViT] Loaded checkpoint from {self.checkpoint_path}")

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.backbone.blocks

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> None:
        self.model(x)
