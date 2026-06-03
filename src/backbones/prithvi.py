"""Prithvi-EO-2.0 backbone adapter (6-channel, single-frame)."""

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import transforms

from src.backbones.base import BackboneAdapter, BackboneSpec


class PrithviEncoder(nn.Module):
    """Lightweight Prithvi-style ViT encoder for single-frame 6-channel input."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        patch_size: int = 16,
        image_size: int = 224,
        in_chans: int = 6,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            img_size=image_size,
            in_chans=in_chans,
            dynamic_img_size=True,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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


class PrithviAdapter(BackboneAdapter):
    def __init__(
        self,
        spec: BackboneSpec,
        device: str = "cuda",
        model_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        allow_rgb_proxy: bool = False,
    ):
        super().__init__(spec, device)
        self.model_id = model_id
        self.allow_rgb_proxy = allow_rgb_proxy
        self.model = nn.Identity()

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.spec.image_size, self.spec.image_size)),
                transforms.ToTensor(),
                PrithviRGBToSixBand(
                    mean=self.spec.normalize_mean,
                    std=self.spec.normalize_std,
                    allow_rgb_proxy=self.allow_rgb_proxy,
                ),
            ]
        )

    def load(self) -> None:
        self.model = PrithviEncoder(
            hidden_dim=self.spec.hidden_dim,
            num_layers=self.spec.num_layers,
            patch_size=self.spec.patch_size,
            image_size=self.spec.image_size,
            in_chans=self.spec.num_channels,
        )
        try:
            ckpt_path = hf_hub_download(repo_id=self.model_id, filename="Prithvi_EO_V2_300M.pt")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            cleaned = {}
            for k, v in state.items():
                key = k.replace("module.", "")
                if any(
                    skip in key
                    for skip in ("decoder", "mask_token", "pred", "neck", "fc_norm")
                ):
                    continue
                cleaned[key.replace("encoder.", "")] = v
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            print(
                f"[Prithvi] Loaded {self.model_id} missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pretrained Prithvi weights from {self.model_id}. "
                "Experiments require real pretrained weights, not a random encoder. "
                f"Original error: {exc}"
            ) from exc
        self.model.eval()

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.backbone.blocks


class PrithviRGBToSixBand:
    """
    Prithvi expects 6 spectral bands; HiRISE v3.2 is RGB only.
    Disabled by default — enable only with --allow-prithvi-rgb-proxy (documented caveat).
    """

    def __init__(self, mean, std, allow_rgb_proxy: bool = False):
        if not allow_rgb_proxy:
            raise RuntimeError(
                "Prithvi requires 6-band input but HiRISE is RGB-only. "
                "Use RGB-native backbones, or pass --allow-prithvi-rgb-proxy to "
                "duplicate RGB channels into 6 bands (not true multispectral data)."
            )
        self.mean = torch.tensor(mean, dtype=torch.float32).view(6, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(6, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != 3:
            bands = x
        else:
            bands = torch.stack([x[0], x[1], x[2], x[1], x[2], x[0]], dim=0)
        return (bands - self.mean) / self.std.clamp_min(1e-6)
