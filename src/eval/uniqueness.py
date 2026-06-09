"""
ECCU Uniqueness Score (ported from Colab DoMars16k notebook).

U(f, x) = alpha * norm_act(f, x) + (1 - alpha) * rarity(f)

  norm_act(f, x) = act(f, x) / global_max(f), clamped to [0, 1]
  rarity(f)      = 1 - c_cond(f)
  c_cond(f)      = mean_C [ count(f activates in C) / |C| ]

A feature activates on an image when activation > activation_threshold.
Live latent: max activation across images > live_eps (default 1e-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _as_labels(labels: Sequence) -> List[str]:
    return [str(lab) for lab in labels]


@dataclass
class UniquenessResult:
    mean_uniqueness: float
    per_latent_uniqueness: np.ndarray
    per_image_scores: np.ndarray
    live_mask: np.ndarray
    n_live: int
    c_cond: np.ndarray
    global_max: np.ndarray


class UniquenessScorer:
    """
    Accumulates dataset statistics via update(), then scores per (feature, image).

    Call update() on the full corpus used to estimate global_max and c_cond
    (Colab used train; fair script may pass test — caller decides).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        activation_threshold: float = 0.0,
        dict_size: int | None = None,
    ):
        self.alpha = float(alpha)
        self.activation_threshold = float(activation_threshold)
        self.dict_size = dict_size

        self._global_max: torch.Tensor | None = None
        self._class_sizes: Dict[str, int] = {}
        self._activate_counts: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self._global_max = None
        self._class_sizes.clear()
        self._activate_counts.clear()

    def update(self, activations: ArrayLike, class_labels: Sequence[str]) -> None:
        """
        Accumulate per-class activation counts and global max per feature.

        activations: (B, dict_size) — max-pooled SAE activations per image.
        class_labels: length B.
        """
        acts = _as_tensor(activations)
        if acts.ndim != 2:
            raise ValueError(f"activations must be 2D (B, dict_size); got {acts.shape}")
        labels = _as_labels(class_labels)
        if acts.shape[0] != len(labels):
            raise ValueError(
                f"activations rows {acts.shape[0]} != labels {len(labels)}"
            )

        n_features = acts.shape[1]
        if self.dict_size is None:
            self.dict_size = n_features
        elif n_features != self.dict_size:
            raise ValueError(f"dict_size mismatch: {n_features} vs {self.dict_size}")

        batch_max = acts.max(dim=0).values
        if self._global_max is None:
            self._global_max = batch_max.clone()
        else:
            self._global_max = torch.maximum(self._global_max, batch_max)

        active = acts > self.activation_threshold
        for i, lab in enumerate(labels):
            self._class_sizes[lab] = self._class_sizes.get(lab, 0) + 1
            if lab not in self._activate_counts:
                self._activate_counts[lab] = torch.zeros(n_features, dtype=torch.float32)
            self._activate_counts[lab] += active[i].float()

    def compute_c_conditional(self) -> torch.Tensor:
        """Return c_cond(f) with shape (dict_size,)."""
        if not self._class_sizes:
            raise RuntimeError("call update() before compute_c_conditional()")
        if self.dict_size is None:
            raise RuntimeError("dict_size unknown; call update() first")

        c_cond = torch.zeros(self.dict_size, dtype=torch.float32)
        n_classes = len(self._class_sizes)
        for lab, n_c in self._class_sizes.items():
            if n_c <= 0:
                continue
            frac = self._activate_counts[lab] / float(n_c)
            c_cond += frac
        c_cond /= float(n_classes)
        return c_cond

    def score(
        self,
        activations: ArrayLike,
        class_labels: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """
        Return uniqueness scores with shape (B, dict_size).

        class_labels is accepted for Colab API parity; not used in the formula
        once c_cond and global_max are fixed from update().
        """
        if self._global_max is None:
            raise RuntimeError("call update() before score()")

        acts = _as_tensor(activations)
        if acts.ndim != 2:
            raise ValueError(f"activations must be 2D (B, dict_size); got {acts.shape}")

        c_cond = self.compute_c_conditional()
        rarity = 1.0 - c_cond

        denom = self._global_max.clamp_min(1e-12)
        norm_act = (acts / denom.unsqueeze(0)).clamp(0.0, 1.0)

        uniqueness = self.alpha * norm_act + (1.0 - self.alpha) * rarity.unsqueeze(0)
        return uniqueness

    @property
    def global_max(self) -> torch.Tensor | None:
        return self._global_max


def live_latent_mask(
    activations: ArrayLike,
    live_eps: float = 1e-8,
) -> np.ndarray:
    """
    Live = max activation across images > live_eps.

    activations: (n_images, dict_size) or (dict_size, n_images).
    Returns boolean (dict_size,).
    """
    acts = _as_tensor(activations)
    if acts.ndim != 2:
        raise ValueError(f"activations must be 2D; got {acts.shape}")
    if acts.shape[0] < acts.shape[1]:
        # Heuristic: fewer rows than cols → likely (n_images, dict_size)
        per_latent_max = acts.max(dim=0).values
    else:
        per_latent_max = acts.max(dim=1).values
    return (per_latent_max > live_eps).cpu().numpy()


def activations_image_major(activations: ArrayLike) -> np.ndarray:
    """Ensure shape (n_images, dict_size)."""
    arr = _as_tensor(activations).cpu().numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activations; got {arr.shape}")
    # Stored matrices in this repo are (dict_size, n_images)
    if arr.shape[0] < arr.shape[1]:
        return arr
    return arr.T


def mean_uniqueness_live(
    activations: ArrayLike,
    class_labels: Sequence[str],
    alpha: float = 0.5,
    activation_threshold: float = 0.0,
    live_eps: float = 1e-8,
    update_activations: ArrayLike | None = None,
    update_labels: Sequence[str] | None = None,
) -> UniquenessResult:
    """
    Full pipeline: update stats, score all images, mean U(f) over live latents.

    Per-feature U(f) = mean_x U(f, x) over images in the scored set.
    update_* defaults to the same activations/labels as the scored set.
    """
    acts_img = activations_image_major(activations)
    labels = _as_labels(class_labels)
    if acts_img.shape[0] != len(labels):
        raise ValueError(
            f"activations images {acts_img.shape[0]} != labels {len(labels)}"
        )

    upd_acts = (
        activations_image_major(update_activations)
        if update_activations is not None
        else acts_img
    )
    upd_labels = _as_labels(update_labels) if update_labels is not None else labels
    if upd_acts.shape[0] != len(upd_labels):
        raise ValueError(
            f"update activations images {upd_acts.shape[0]} != labels {len(upd_labels)}"
        )

    scorer = UniquenessScorer(
        alpha=alpha,
        activation_threshold=activation_threshold,
        dict_size=acts_img.shape[1],
    )
    scorer.update(upd_acts, upd_labels)
    scores = scorer.score(acts_img, labels).cpu().numpy()
    c_cond = scorer.compute_c_conditional().cpu().numpy()
    gmax = scorer.global_max.cpu().numpy() if scorer.global_max is not None else np.array([])

    live = live_latent_mask(acts_img, live_eps=live_eps)
    per_latent = scores.mean(axis=0)
    if not live.any():
        mean_u = float("nan")
    else:
        mean_u = float(per_latent[live].mean())

    return UniquenessResult(
        mean_uniqueness=mean_u,
        per_latent_uniqueness=per_latent,
        per_image_scores=scores,
        live_mask=live,
        n_live=int(live.sum()),
        c_cond=c_cond,
        global_max=gmax,
    )


def random_baseline_mean(
    activations: ArrayLike,
    class_labels: Sequence[str],
    alpha: float = 0.5,
    activation_threshold: float = 0.0,
    live_eps: float = 1e-8,
    seed: int = 42,
) -> float:
    """Shuffle labels, recompute mean uniqueness (same images, permuted classes)."""
    labels = _as_labels(class_labels)
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    result = mean_uniqueness_live(
        activations=activations,
        class_labels=shuffled,
        alpha=alpha,
        activation_threshold=activation_threshold,
        live_eps=live_eps,
        update_activations=activations,
        update_labels=shuffled,
    )
    return result.mean_uniqueness


def beats_random(
    real_mean: float,
    shuffled_mean: float,
) -> bool:
    if np.isnan(real_mean) or np.isnan(shuffled_mean):
        return False
    return real_mean > shuffled_mean
