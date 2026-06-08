#!/usr/bin/env python3
"""
HTML visual inspection report: top-activating test images per SAE latent.

Example:
  python scripts/visualize_top_activations.py \\
    --backbone momo --layer middle --sae-arch topk \\
    --tag middle_topk_d1_perdim_aux1p0_full \\
    --output results/visualizations/momo_middle_topk_top_activations.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hirise import (  # noqa: E402
    filter_readable_official_samples,
    load_official_split_table,
    load_readable_cache,
)
from src.eval.fair_extra_metrics import _mean_pooled_backbone_embeddings, _pairwise_cosine_mean
from src.eval.saebench_interpretability import build_image_activation_matrix
from src.eval.saebench_sparse_probing import label_display_name
from src.paths import RESULTS_DIR
from src.sae.activation_norm import norm_spec_for_eval
from src.sae.trainer import load_sae_checkpoint

DEFAULT_LABELS = Path(
    r"G:\My Drive\metrics&models\mars_data\hirise_v3_2\labels-map-proj-v3_2.txt"
)
DEFAULT_IMAGES = Path(r"D:\hirise_v3_2\images")
THUMB_SIZE = 128
PLACEHOLDER_JPEG: bytes | None = None


@dataclass
class LatentReport:
    latent_idx: int
    mono_ms_score: float
    label_purity: float
    majority_class: str
    top_images: List[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--sae-arch", required=True)
    parser.add_argument("--tag", required=True, help="Ablation tag under results/ablations/<backbone>/")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    parser.add_argument("--test-cache", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--readable-cache", type=Path, default=RESULTS_DIR / "hirise_readable_cache.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-n-latents", type=int, default=50)
    parser.add_argument("--top-n-images", type=int, default=16)
    parser.add_argument("--min-live-images", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-chunk", type=int, default=512)
    return parser.parse_args()


def resolve_test_cache(args: argparse.Namespace) -> Path:
    if args.test_cache is not None:
        if not args.test_cache.is_file():
            raise FileNotFoundError(f"Missing --test-cache {args.test_cache}")
        return args.test_cache

    candidates = [
        args.cache_dir / f"{args.backbone}_{args.layer}_test_activations.pt",
        args.results_dir
        / "probe_activation_cache"
        / args.backbone
        / args.layer
        / "eval_test_activations.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No test activation cache found. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    path = (
        args.results_dir
        / "ablations"
        / args.backbone
        / args.tag
        / args.backbone
        / args.layer
        / args.sae_arch
        / "sae_weights.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}")
    return path


def load_test_samples(args: argparse.Namespace) -> list[dict]:
    cached = load_readable_cache(args.readable_cache)
    if cached is None:
        raise FileNotFoundError(f"Missing readable cache {args.readable_cache}")
    readable = set(cached.get("readable_rel_paths", []))
    split_rows = load_official_split_table(args.labels_file)
    samples = filter_readable_official_samples(
        split_rows, readable, ["test"], landforms_only=True
    )
    if not samples:
        raise RuntimeError("No readable official test samples found")
    return samples


def load_test_cache(path: Path) -> tuple[torch.Tensor, list[str], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    acts = payload["activations"].float()
    labels = [str(x) for x in payload["labels"]]
    n_images = int(payload.get("n_images", len(labels)))
    return acts, labels, n_images


def patches_per_image(n_patches: int, n_images: int) -> int:
    if n_patches % n_images != 0:
        raise ValueError(
            f"Activation rows {n_patches} not divisible by n_images={n_images}"
        )
    return n_patches // n_images


def gray_placeholder_b64() -> str:
    global PLACEHOLDER_JPEG
    if PLACEHOLDER_JPEG is None:
        img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), color=(160, 160, 160))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        PLACEHOLDER_JPEG = buf.getvalue()
    return base64.b64encode(PLACEHOLDER_JPEG).decode("ascii")


def image_to_b64(path: Path) -> str:
    if not path.is_file():
        return gray_placeholder_b64()
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (32, 32, 32))
            x = (THUMB_SIZE - im.width) // 2
            y = (THUMB_SIZE - im.height) // 2
            canvas.paste(im, (x, y))
            buf = io.BytesIO()
            canvas.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except OSError:
        return gray_placeholder_b64()


def latent_purity(labels: list[str], top_idx: np.ndarray) -> tuple[float, str]:
    top_labels = [labels[i] for i in top_idx]
    counts = Counter(top_labels)
    majority_label, majority_count = counts.most_common(1)[0]
    purity = majority_count / len(top_idx)
    return float(purity), label_display_name(majority_label)


def build_latent_reports(
    matrix: np.ndarray,
    x_bb: np.ndarray,
    labels: list[str],
    samples: list[dict],
    images_dir: Path,
    top_n_images: int,
    min_live_images: int,
) -> list[LatentReport]:
    dict_size, n_images = matrix.shape
    k = min(top_n_images, n_images)
    reports: list[LatentReport] = []

    for j in range(dict_size):
        active_count = int((matrix[j] > 0).sum())
        if active_count < min_live_images:
            continue
        top_idx = np.argsort(-matrix[j])[:k]
        mono_ms = _pairwise_cosine_mean(x_bb[top_idx])
        purity, majority = latent_purity(labels, top_idx)

        top_images = []
        for rank, img_i in enumerate(top_idx, start=1):
            rel = samples[img_i]["rel_path"]
            path = images_dir / rel
            top_images.append(
                {
                    "rank": rank,
                    "img_i": int(img_i),
                    "rel_path": rel,
                    "filename": Path(rel).name,
                    "label": label_display_name(labels[img_i]),
                    "activation": float(matrix[j, img_i]),
                    "b64": image_to_b64(path),
                }
            )

        reports.append(
            LatentReport(
                latent_idx=j,
                mono_ms_score=float(mono_ms),
                label_purity=purity,
                majority_class=majority,
                top_images=top_images,
            )
        )
    reports.sort(key=lambda r: r.mono_ms_score, reverse=True)
    return reports


def render_html(
    args: argparse.Namespace,
    reports: list[LatentReport],
    summary: dict,
) -> str:
    sections = []
    for r in reports[: args.top_n_latents]:
        cells = []
        for img in r.top_images:
            cap = (
                f"{html.escape(img['filename'])}<br>"
                f"{html.escape(img['label'])}<br>"
                f"act={img['activation']:.4f}"
            )
            cells.append(
                f'<div class="cell">'
                f'<img src="data:image/jpeg;base64,{img["b64"]}" alt="{html.escape(img["filename"])}"/>'
                f'<div class="cap">{cap}</div></div>'
            )
        while len(cells) < args.top_n_images:
            cells.append(
                f'<div class="cell empty"><div class="cap">—</div></div>'
            )

        sections.append(
            f"""
<section class="latent">
  <h2>Latent {r.latent_idx}</h2>
  <div class="meta">
    <span><b>MS score</b> {r.mono_ms_score:.4f}</span>
    <span><b>Label purity</b> {r.label_purity:.2f}</span>
    <span><b>Majority class</b> {html.escape(r.majority_class)}</span>
  </div>
  <div class="grid">{''.join(cells)}</div>
</section>
"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(summary['title'])}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
h1 {{ margin-bottom: 8px; }}
.summary {{ background: #1c1c1c; padding: 12px 16px; border-radius: 8px; margin-bottom: 24px; }}
.summary span {{ margin-right: 18px; }}
.latent {{ margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid #333; }}
.latent h2 {{ margin: 0 0 8px; }}
.meta span {{ margin-right: 16px; color: #bbb; }}
.grid {{ display: grid; grid-template-columns: repeat(4, {THUMB_SIZE}px); gap: 10px; }}
.cell {{ width: {THUMB_SIZE}px; }}
.cell img {{ width: {THUMB_SIZE}px; height: {THUMB_SIZE}px; display: block; border-radius: 4px; }}
.cap {{ font-size: 11px; line-height: 1.3; margin-top: 4px; color: #aaa; word-break: break-all; }}
.cell.empty {{ height: {THUMB_SIZE}px; background: #222; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{html.escape(summary['title'])}</h1>
<div class="summary">
  <span><b>Backbone</b> {html.escape(summary['backbone'])}</span>
  <span><b>Layer</b> {html.escape(summary['layer'])}</span>
  <span><b>Arch</b> {html.escape(summary['sae_arch'])}</span>
  <span><b>Dict size</b> {summary['dict_size']}</span>
  <span><b>Live latents</b> {summary['live_latents']}</span>
  <span><b>Dead latents</b> {summary['dead_latents']}</span>
  <span><b>Test images</b> {summary['n_test']}</span>
  <span><b>Shown latents</b> {min(args.top_n_latents, len(reports))} / {summary['live_latents']}</span>
</div>
{''.join(sections)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    checkpoint = resolve_checkpoint(args)
    cache_path = resolve_test_cache(args)
    output = args.output or (
        args.results_dir
        / "visualizations"
        / f"{args.backbone}_{args.layer}_{args.sae_arch}_top_activations.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    samples = load_test_samples(args)
    test_acts, cache_labels, n_images = load_test_cache(cache_path)
    if n_images != len(samples):
        raise ValueError(
            f"Cache n_images={n_images} != readable test samples={len(samples)}"
        )
    sample_labels = [s["label"] for s in samples]
    if cache_labels != sample_labels:
        tqdm.write(
            "[WARN] Cache label order differs from official test manifest; "
            "using cache labels for purity, manifest paths for images."
        )
        labels = cache_labels
    else:
        labels = sample_labels

    ppi = patches_per_image(test_acts.shape[0], n_images)
    sae, payload = load_sae_checkpoint(checkpoint, args.device)
    norm_spec = norm_spec_for_eval(payload)
    dict_size = sae.dict_size if hasattr(sae, "dict_size") else sae.encoder.out_features

    tqdm.write(f"Encoding {n_images} test images from {cache_path.name}...")
    matrix = build_image_activation_matrix(
        sae,
        test_acts,
        n_images,
        norm_spec,
        args.device,
        encode_chunk=args.encode_chunk,
    )
    x_bb = _mean_pooled_backbone_embeddings(test_acts, n_images, ppi, norm_spec)

    live_mask = np.array([(matrix[j] > 0).sum() >= args.min_live_images for j in range(dict_size)])
    live_latents = int(live_mask.sum())
    dead_latents = int(dict_size - live_latents)

    tqdm.write("Building latent reports and embedding thumbnails...")
    reports = build_latent_reports(
        matrix=matrix,
        x_bb=x_bb,
        labels=labels,
        samples=samples,
        images_dir=args.images_dir,
        top_n_images=args.top_n_images,
        min_live_images=args.min_live_images,
    )

    summary = {
        "title": f"{args.backbone} {args.layer} {args.sae_arch} — top activations",
        "backbone": args.backbone,
        "layer": args.layer,
        "sae_arch": args.sae_arch,
        "dict_size": dict_size,
        "live_latents": live_latents,
        "dead_latents": dead_latents,
        "n_test": n_images,
    }
    html_text = render_html(args, reports, summary)
    output.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output} ({len(reports)} live latents, showing top {min(args.top_n_latents, len(reports))})")


if __name__ == "__main__":
    main()
