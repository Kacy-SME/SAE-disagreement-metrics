"""Backbone registry and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import yaml

from src.backbones.base import BackboneAdapter, BackboneSpec, layer_indices_for_backbone
from src.paths import CONFIGS_DIR, DEFAULT_PATHS

if TYPE_CHECKING:
    pass


def _import_adapter(loader: str):
    if loader == "timm":
        from src.backbones.timm_backbone import TimmBackboneAdapter

        return TimmBackboneAdapter
    if loader == "momo":
        from src.backbones.momo import MOMOAdapter

        return MOMOAdapter
    if loader == "mars_orbital_vit":
        from src.backbones.mars_vit import MarsOrbitalViTAdapter

        return MarsOrbitalViTAdapter
    if loader == "prithvi":
        from src.backbones.prithvi import PrithviAdapter

        return PrithviAdapter
    if loader == "croma":
        from src.backbones.croma import CROMAAdapter

        return CROMAAdapter
    if loader == "satmae_pp":
        from src.backbones.satmae_pp import SatMAEppAdapter

        return SatMAEppAdapter
    raise ValueError(f"Unsupported loader {loader!r}")


def load_backbone_configs(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    cfg_path = path or (CONFIGS_DIR / "backbones.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["backbones"]


def build_backbone_spec(name: str, cfg: Dict[str, Any]) -> BackboneSpec:
    norm = cfg.get("normalize", {})
    return BackboneSpec(
        name=name,
        display_name=cfg.get("display_name", name),
        num_layers=int(cfg["num_layers"]),
        patch_size=int(cfg["patch_size"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_prefix_tokens=int(cfg.get("num_prefix_tokens", 1)),
        image_size=int(cfg.get("image_size", 224)),
        normalize_mean=list(norm.get("mean", [0.485, 0.456, 0.406])),
        normalize_std=list(norm.get("std", [0.229, 0.224, 0.225])),
        num_channels=int(cfg.get("num_channels", 3)),
    )


def resolve_checkpoint(name: str, cfg: Dict[str, Any]) -> str | None:
    if cfg.get("checkpoint_path"):
        return cfg["checkpoint_path"]
    if name == "mars_orbital_vit":
        return DEFAULT_PATHS["mars_orbital_vit_checkpoint"]
    if name == "momo" and DEFAULT_PATHS.get("momo_checkpoint"):
        return DEFAULT_PATHS["momo_checkpoint"]
    return None


def create_backbone(
    name: str,
    device: str = "cuda",
    backbone_cfg: Dict[str, Any] | None = None,
    configs: Dict[str, Dict[str, Any]] | None = None,
    allow_prithvi_rgb_proxy: bool = False,
) -> BackboneAdapter:
    configs = configs or load_backbone_configs()
    if name not in configs:
        raise KeyError(f"Unknown backbone {name!r}; available: {list(configs)}")
    cfg = dict(configs[name])
    if backbone_cfg:
        cfg.update(backbone_cfg)
    spec = build_backbone_spec(name, cfg)
    loader = cfg.get("loader", "timm")
    checkpoint = resolve_checkpoint(name, cfg)
    AdapterCls = _import_adapter(loader)

    if loader == "timm":
        adapter = AdapterCls(
            spec=spec,
            timm_model=cfg["timm_model"],
            device=device,
            checkpoint_path=checkpoint,
        )
    elif loader == "momo":
        adapter = AdapterCls(
            spec=spec,
            timm_model=cfg.get("timm_model", "vit_base_patch16_224"),
            device=device,
            checkpoint_path=checkpoint,
            hf_repo=cfg.get("model_id", "Mirali33/MOMO"),
            hf_filename=cfg.get("checkpoint_filename", "vit-b-16/momo.pth"),
        )
    elif loader == "mars_orbital_vit":
        adapter = AdapterCls(
            spec=spec,
            device=device,
            checkpoint_path=checkpoint,
            num_register_tokens=int(cfg.get("num_register_tokens", 4)),
            timm_model=cfg.get("timm_model", "vit_base_patch14_dinov2"),
        )
    elif loader == "prithvi":
        adapter = AdapterCls(
            spec=spec,
            device=device,
            model_id=cfg.get("model_id", "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"),
            allow_rgb_proxy=allow_prithvi_rgb_proxy,
        )
    elif loader == "croma":
        adapter = AdapterCls(
            spec=spec,
            device=device,
            hf_repo=cfg.get("model_id", "antofuller/CROMA"),
            hf_filename=cfg.get("checkpoint_filename", "CROMA_base.pt"),
            checkpoint_path=checkpoint,
        )
    elif loader == "satmae_pp":
        adapter = AdapterCls(
            spec=spec,
            device=device,
            timm_model=cfg.get("timm_model", "vit_large_patch16_224"),
            hf_repo=cfg.get("model_id", "mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb"),
            hf_filename=cfg.get(
                "checkpoint_filename", "checkpoint_ViT-L_pretrain_fmow_rgb.pth"
            ),
            checkpoint_path=checkpoint,
        )
    else:
        raise ValueError(f"Unsupported loader {loader!r} for backbone {name!r}")

    adapter.load()
    return adapter.to(device)


def list_experiment_combinations(
    backbone_names: List[str] | None = None,
    layer_depths: List[str] | None = None,
    sae_archs: List[str] | None = None,
    configs: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    configs = configs or load_backbone_configs()
    backbone_names = backbone_names or list(configs.keys())
    layer_depths = layer_depths or ["early", "middle", "late"]
    sae_archs = sae_archs or ["topk", "matryoshka"]

    combos: List[Dict[str, Any]] = []
    for backbone in backbone_names:
        n_layers = int(configs[backbone]["num_layers"])
        layer_map = layer_indices_for_backbone(n_layers)
        for depth in layer_depths:
            for sae_arch in sae_archs:
                combos.append(
                    {
                        "backbone": backbone,
                        "display_name": configs[backbone].get("display_name", backbone),
                        "layer_depth": depth,
                        "layer_index": layer_map[depth],
                        "sae_arch": sae_arch,
                        "hidden_dim": int(configs[backbone]["hidden_dim"]),
                    }
                )
    return combos
