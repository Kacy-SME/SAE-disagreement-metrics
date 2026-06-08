"""Patch-token activation extraction via forward hooks."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.backbones.base import BackboneAdapter


class ActivationHook:
    def __init__(self, num_prefix_tokens: int):
        self.num_prefix_tokens = num_prefix_tokens
        self.storage: List[torch.Tensor] = []
        self._handle = None

    def _hook(self, module, inputs, output):
        if output.ndim == 3:
            patch_tokens = output[:, self.num_prefix_tokens :, :]
            flat = patch_tokens.reshape(-1, patch_tokens.shape[-1])
            self.storage.append(flat.detach().cpu())

    def register(self, block: torch.nn.Module):
        self._handle = block.register_forward_hook(self._hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def drain(self) -> torch.Tensor:
        if not self.storage:
            return torch.empty(0, dtype=torch.float32)
        out = torch.cat(self.storage, dim=0)
        self.storage.clear()
        return out


def _register_cls_zero_pre_hook(
    blocks: torch.nn.ModuleList,
    zero_cls_token: bool,
) -> Optional[torch.utils.hooks.RemovableHandle]:
    """Zero CLS (first prefix token) before the first transformer block."""
    if not zero_cls_token or len(blocks) == 0:
        return None

    def pre_hook(module, args):
        x = args[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return args
        x = x.clone()
        x[:, :1, :] = 0
        return (x,) + args[1:]

    return blocks[0].register_forward_pre_hook(pre_hook)


@torch.no_grad()
def extract_patch_activations(
    backbone: BackboneAdapter,
    dataloader: DataLoader,
    layer_index: int,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    desc: str = "Extract activations",
    zero_cls_token: bool = False,
) -> torch.Tensor:
    backbone.model.eval()
    hook = ActivationHook(backbone.spec.num_prefix_tokens)
    blocks = backbone.blocks
    if layer_index < 0 or layer_index >= len(blocks):
        raise IndexError(
            f"layer_index={layer_index} out of range for {len(blocks)} blocks"
        )
    cls_hook = _register_cls_zero_pre_hook(blocks, zero_cls_token)
    hook.register(blocks[layer_index])

    total = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    batch_count = 0
    pbar = tqdm(dataloader, total=total, desc=desc, unit="batch", leave=False)
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, non_blocking=True)
        backbone.forward(images)
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    pbar.close()
    hook.remove()
    if cls_hook is not None:
        cls_hook.remove()
    out = hook.drain()
    if out.numel() == 0:
        raise RuntimeError(
            f"{desc}: no patch activations collected (empty dataloader or forward failed)."
        )
    return out


@torch.no_grad()
def stream_patch_activations_to_buffer(
    backbone: BackboneAdapter,
    dataloader: DataLoader,
    layer_index: int,
    buffer: "ActivationBuffer",
    device: str = "cuda",
    max_batches: Optional[int] = None,
    desc: str = "Stream activations",
) -> float:
    """
    Extract patch activations batch-by-batch into buffer (bounded RAM).
    Returns norm_scalar = sqrt(mean(||x||^2)) over all streamed vectors.
    """
    backbone.model.eval()
    hook = ActivationHook(backbone.spec.num_prefix_tokens)
    blocks = backbone.blocks
    if layer_index < 0 or layer_index >= len(blocks):
        raise IndexError(
            f"layer_index={layer_index} out of range for {len(blocks)} blocks"
        )
    hook.register(blocks[layer_index])

    total_sq_norm = 0.0
    vector_count = 0
    total = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    batch_count = 0

    pbar = tqdm(dataloader, total=total, desc=desc, unit="batch", leave=False)
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, non_blocking=True)
        backbone.forward(images)
        batch_acts = hook.drain()
        if batch_acts.numel() > 0:
            buffer.add(batch_acts)
            total_sq_norm += batch_acts.pow(2).sum(dim=-1).sum().item()
            vector_count += batch_acts.shape[0]
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    pbar.close()
    hook.remove()

    if vector_count == 0:
        raise RuntimeError(f"{desc}: no patch activations collected.")
    return float((total_sq_norm / vector_count) ** 0.5)


class ActivationBuffer:
    """
    Reservoir of patch activation vectors for SAE training.
    Samples randomly with replacement from a fixed-capacity pool.

    Refill policy (buffer_refill_threshold in config):
      0  — fill once at start, then never refill (recommended for full HiRISE runs).
      >0 — also refill every N SAE steps where N = int(capacity * threshold / 1024)
           (optional freshness; each refill re-streams the train loader through ViT).
    """

    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        refill_threshold: float = 0.0,
        sae_batch_size: int = 1024,
    ):
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.refill_threshold = refill_threshold
        self.sae_batch_size = sae_batch_size
        self.storage = torch.empty(0, hidden_dim)
        self._last_refill_step = -1
        if refill_threshold > 0:
            self._refill_every_steps = max(
                1, int(capacity * refill_threshold / sae_batch_size)
            )
        else:
            self._refill_every_steps = 0

    def __len__(self) -> int:
        return int(self.storage.shape[0])

    def needs_refill(self, training_step: int = 0) -> bool:
        if self.storage.shape[0] < self.sae_batch_size:
            return True
        if self._refill_every_steps <= 0:
            return False
        if self._last_refill_step < 0:
            return False
        return (training_step - self._last_refill_step) >= self._refill_every_steps

    def mark_refilled(self, training_step: int = 0) -> None:
        self._last_refill_step = training_step

    def add(self, vectors: torch.Tensor) -> None:
        vectors = vectors.detach().cpu().float()
        if vectors.numel() == 0:
            return
        if self.storage.numel() == 0:
            self.storage = vectors
        else:
            self.storage = torch.cat([self.storage, vectors], dim=0)
        if self.storage.shape[0] > self.capacity:
            idx = torch.randperm(self.storage.shape[0])[: self.capacity]
            self.storage = self.storage[idx]

    def sample(self, batch_size: int) -> torch.Tensor:
        n = self.storage.shape[0]
        if n < batch_size:
            raise RuntimeError(
                f"Buffer underflow: need {batch_size} vectors, buffer has {n}. "
                "Refill the buffer before sampling."
            )
        idx = torch.randint(0, n, (batch_size,))
        return self.storage[idx]


def activation_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
