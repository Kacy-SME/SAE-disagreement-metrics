"""TopK sparse autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sae.utils import normalize_decoder_columns


@dataclass
class SAELossOut:
    total: torch.Tensor
    recon: torch.Tensor
    aux: torch.Tensor
    l0: torch.Tensor


class TopKSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        dict_size: int,
        k: int = 40,
        k_aux: int = 384,
    ):
        super().__init__()
        self.d_in = d_in
        self.dict_size = dict_size
        self.k = k
        self.k_aux = k_aux
        self.encoder = nn.Linear(d_in, dict_size)
        self.decoder = nn.Linear(dict_size, d_in, bias=True)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer(
            "stats_last_nonzero",
            torch.zeros(dict_size, dtype=torch.long),
        )
        self._step = 0

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.b_dec
        pre_acts = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts, pre_acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acts, pre_acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts, pre_acts

    def _auxk_loss(
        self,
        x: torch.Tensor,
        acts: torch.Tensor,
        pre_acts: torch.Tensor,
    ) -> torch.Tensor:
        residual = x - self.decode(acts)
        dead_mask = self.stats_last_nonzero == 0
        if not dead_mask.any():
            return torch.zeros((), device=x.device, dtype=x.dtype)
        k_aux = min(self.k_aux, int(dead_mask.sum().item()))
        if k_aux == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        dead_pre = pre_acts[:, dead_mask]
        aux_vals, aux_idx = torch.topk(dead_pre, k_aux, dim=-1)
        aux_acts = torch.zeros_like(pre_acts[:, dead_mask])
        aux_acts.scatter_(-1, aux_idx, F.relu(aux_vals))
        full_aux = torch.zeros_like(pre_acts)
        full_aux[:, dead_mask] = aux_acts
        aux_recon = self.decode(acts + full_aux)
        return F.mse_loss(aux_recon, x)

    def loss(self, x: torch.Tensor, aux_weight: float = 1.0) -> SAELossOut:
        recon, acts, pre_acts = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        aux_loss = self._auxk_loss(x, acts, pre_acts)
        l0 = (acts > 0).float().sum(dim=-1).mean()
        total = recon_loss + aux_weight * aux_loss
        return SAELossOut(total=total, recon=recon_loss, aux=aux_loss, l0=l0)

    @torch.no_grad()
    def update_dead_latent_stats(self, acts: torch.Tensor) -> None:
        fired = (acts > 0).any(dim=0)
        self.stats_last_nonzero[fired] = self._step
        self._step += 1

    def post_grad_step(self) -> None:
        normalize_decoder_columns(self.decoder)

    def config_dict(self) -> Dict:
        return {
            "arch": "topk",
            "d_in": self.d_in,
            "dict_size": self.dict_size,
            "k": self.k,
            "k_aux": self.k_aux,
        }
