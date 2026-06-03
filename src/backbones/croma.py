"""CROMA optical encoder backbone."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from src.backbones.base import BackboneAdapter, BackboneSpec


class CROMAOpticalEncoder(nn.Module):
    """Minimal CROMA optical ViT encoder for patch-token extraction."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        patch_size: int = 16,
        image_size: int = 120,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            img_size=image_size,
            dynamic_img_size=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        if x.ndim == 4:
            b, h, w, c = x.shape
            x = x.reshape(b, h * w, c)
        cls = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.backbone.pos_embed is not None:
            pos = self.backbone.pos_embed
            if pos.shape[1] == x.shape[1]:
                x = x + pos
            else:
                x[:, :1] = x[:, :1] + pos[:, :1]
                x[:, 1:] = x[:, 1:] + pos[:, 1 : x.shape[1]]
        x = self.backbone.pos_drop(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        return self.backbone.norm(x)


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
        self.model = nn.Identity()

    def _resolve_checkpoint(self) -> str:
        if self.checkpoint_path:
            return self.checkpoint_path
        return hf_hub_download(repo_id=self.hf_repo, filename=self.hf_filename)

    def load(self) -> None:
        self.model = CROMAOpticalEncoder(
            hidden_dim=self.spec.hidden_dim,
            num_layers=self.spec.num_layers,
            patch_size=self.spec.patch_size,
            image_size=self.spec.image_size,
        )
        ckpt_path = self._resolve_checkpoint()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "optical_encoder" in state:
                state = state["optical_encoder"]
            elif "state_dict" in state:
                state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            key = k.replace("module.", "")
            if key.startswith("optical_encoder."):
                key = key[len("optical_encoder.") :]
            cleaned[key] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(
            f"[CROMA] checkpoint={ckpt_path} missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        self.model.eval()

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.backbone.blocks
