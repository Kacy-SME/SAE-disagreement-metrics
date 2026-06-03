"""Unified backbone adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import transforms


@dataclass
class BackboneSpec:
    name: str
    display_name: str
    num_layers: int
    patch_size: int
    hidden_dim: int
    num_prefix_tokens: int
    image_size: int
    normalize_mean: List[float]
    normalize_std: List[float]
    num_channels: int = 3


class BackboneAdapter(ABC):
    def __init__(self, spec: BackboneSpec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.model: nn.Module

    @abstractmethod
    def load(self) -> None:
        ...

    @property
    def blocks(self) -> nn.ModuleList:
        raise NotImplementedError

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.spec.image_size, self.spec.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.spec.normalize_mean,
                    std=self.spec.normalize_std,
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> None:
        self.model(x)

    def layer_index(self, depth: str) -> int:
        n = self.spec.num_layers
        mapping = {
            "early": n // 4,
            "middle": n // 2,
            "late": (3 * n) // 4,
        }
        if depth not in mapping:
            raise ValueError(f"Unknown depth {depth!r}; expected early/middle/late")
        return mapping[depth]

    def to(self, device: str) -> "BackboneAdapter":
        self.device = device
        self.model.to(device)
        self.model.eval()
        return self


def layer_indices_for_backbone(num_layers: int) -> Dict[str, int]:
    return {
        "early": num_layers // 4,
        "middle": num_layers // 2,
        "late": (3 * num_layers) // 4,
    }
