"""Export a small HTML gallery of sample HiRISE post-2025 patches."""

from __future__ import annotations

import argparse
import base64
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.post_may2025 import _scale_to_uint8, scan_patch_cache
from src.paths import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Post-2025 patch cache (default: D:\\hirise_post2025_cache\\patches)",
    )
    parser.add_argument("--n-images", type=int, default=12)
    parser.add_argument("--patches-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "visualizations" / "hirise_post2025_patch_preview.html",
    )
    args = parser.parse_args()

    from src.data.post_may2025 import default_patch_cache_dir

    random.seed(args.seed)
    entries = scan_patch_cache(args.cache_dir or default_patch_cache_dir())
    by_stem: dict[str, list] = {}
    for e in entries:
        by_stem.setdefault(e["image_stem"], []).append(e)
    stems = sorted(by_stem.keys())
    pick_stems = random.sample(stems, min(args.n_images, len(stems)))

    rows = []
    for stem in pick_stems:
        sub = by_stem[stem]
        idxs = random.sample(range(len(sub)), min(args.patches_per_image, len(sub)))
        for i in idxs:
            rows.append(sub[i])

    cards: list[str] = []
    for r in rows:
        arr = np.load(r["patch_path"])
        gray = _scale_to_uint8(arr)
        rgb = np.stack([gray, gray, gray], axis=-1)
        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        title = (
            f"{r['obs_id']} | {r['image_stem']} | patch {int(r['patch_idx'])} | "
            f"y={int(r['y'])} x={int(r['x'])}"
        )
        cards.append(
            f'<div class="card"><img src="data:image/png;base64,{b64}" '
            f'alt="{title}"><p>{title}</p></div>'
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>HiRISE post-2025 patch preview</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 1rem; }}
h1 {{ font-size: 1.2rem; font-weight: 500; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; }}
.card {{ background: #1a1a1a; border-radius: 8px; padding: 0.5rem; }}
.card img {{ width: 100%; height: auto; border-radius: 4px; image-rendering: pixelated; }}
.card p {{ font-size: 0.75rem; color: #aaa; margin: 0.5rem 0 0; word-break: break-all; }}
</style></head><body>
<h1>HiRISE post-2025 — {len(cards)} sample patches from {len(pick_stems)} scenes</h1>
<div class="grid">{"".join(cards)}</div>
</body></html>"""
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
