"""MOMO Mars orbital ViT backbone."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from src.backbones.base import BackboneAdapter, BackboneSpec


class MOMOAdapter(BackboneAdapter):
    def __init__(
        self,
        spec: BackboneSpec,
        timm_model: str,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        hf_repo: str = "Mirali33/MOMO",
        hf_filename: str = "vit-b-16/momo.pth",
    ):
        super().__init__(spec, device)
        self.timm_model = timm_model
        self.checkpoint_path = checkpoint_path
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.model = nn.Identity()

    def _resolve_checkpoint(self) -> str:
        if self.checkpoint_path:
            return self.checkpoint_path
        return hf_hub_download(repo_id=self.hf_repo, filename=self.hf_filename)

    def load(self) -> None:
        self.model = timm.create_model(
            self.timm_model,
            pretrained=False,
            num_classes=0,
            img_size=self.spec.image_size,
        )
        ckpt_path = self._resolve_checkpoint()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            elif "teacher" in state:
                state = state["teacher"]
        cleaned = {}
        for k, v in state.items():
            key = k.replace("module.", "").replace("backbone.", "")
            if key.startswith("head."):
                continue
            cleaned[key] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(
            f"[MOMO] checkpoint={ckpt_path} missing={len(missing)} unexpected={len(unexpected)}"
        )
        self.model.eval()

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.blocks
