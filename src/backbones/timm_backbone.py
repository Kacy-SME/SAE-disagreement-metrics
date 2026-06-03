"""Generic timm ViT backbone adapter."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from src.backbones.base import BackboneAdapter, BackboneSpec


class TimmBackboneAdapter(BackboneAdapter):
    def __init__(
        self,
        spec: BackboneSpec,
        timm_model: str,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__(spec, device)
        self.timm_model = timm_model
        self.checkpoint_path = checkpoint_path
        self.pretrained = pretrained
        self.model = nn.Identity()

    def load(self) -> None:
        self.model = timm.create_model(
            self.timm_model,
            pretrained=self.pretrained and self.checkpoint_path is None,
            num_classes=0,
            img_size=self.spec.image_size,
            dynamic_img_size=True,
        )
        if self.checkpoint_path:
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"[{self.spec.name}] Loaded checkpoint with missing keys: {len(missing)}")
            if unexpected:
                print(f"[{self.spec.name}] Loaded checkpoint with unexpected keys: {len(unexpected)}")
        self.model.eval()

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.blocks
