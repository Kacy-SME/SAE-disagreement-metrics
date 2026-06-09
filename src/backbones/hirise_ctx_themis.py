"""MOMO joint HiRISE+CTX+THEMIS backbone (ViT-B/16, same loader as MOMO)."""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from src.backbones.momo import MOMOAdapter


class HiriseCtxThemisAdapter(MOMOAdapter):
    """
    Loads hirise_ctx_themis.pth with identical timm ViT-B/16 architecture and
    ImageNet normalization as the standard MOMO backbone.
    """

    def _resolve_checkpoint(self) -> str:
        if self.checkpoint_path:
            return self.checkpoint_path
        if self.hf_repo and self.hf_filename:
            return hf_hub_download(repo_id=self.hf_repo, filename=self.hf_filename)
        raise FileNotFoundError(
            "hirise_ctx_themis checkpoint not found. Expected "
            "G:\\My Drive\\metrics&models\\existing_model_checkpoints\\hirise_ctx_themis.pth "
            "or set HIRISE_CTX_THEMIS_CHECKPOINT_PATH."
        )
