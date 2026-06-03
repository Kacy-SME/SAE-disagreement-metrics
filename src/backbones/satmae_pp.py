"""SatMAE++ ViT-L backbone."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from src.backbones.base import BackboneAdapter, BackboneSpec


class SatMAEppAdapter(BackboneAdapter):
    def __init__(
        self,
        spec: BackboneSpec,
        device: str = "cuda",
        timm_model: str = "vit_large_patch16_224",
        hf_repo: str = "mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb",
        hf_filename: str = "checkpoint_ViT-L_pretrain_fmow_rgb.pth",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(spec, device)
        self.timm_model = timm_model
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.checkpoint_path = checkpoint_path
        self.model = nn.Identity()

    def _resolve_checkpoint(self) -> str:
        if self.checkpoint_path:
            return self.checkpoint_path
        return hf_hub_download(
            repo_id=self.hf_repo,
            filename=self.hf_filename,
        )

    def load(self) -> None:
        self.model = timm.create_model(
            self.timm_model,
            pretrained=False,
            num_classes=0,
            img_size=self.spec.image_size,
            dynamic_img_size=True,
        )
        ckpt_path = self._resolve_checkpoint()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            key = k.replace("module.", "")
            if key.startswith("decoder.") or key.startswith("mask_token"):
                continue
            cleaned[key] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(
            f"[SatMAE++] checkpoint={ckpt_path} missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        self.model.eval()

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.blocks
