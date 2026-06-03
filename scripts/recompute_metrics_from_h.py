#!/usr/bin/env python3
"""Print corrected loss_recovered from saved eval_metrics.json H_* fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import compute_loss_recovered  # noqa: E402


def main() -> None:
    root = Path(PROJECT_ROOT / "results")
    for path in sorted(root.rglob("eval_metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        h_with = m["H_with_sae"]
        h_zero = m["H_zero_ablate"]
        h_orig = m["H_original"]
        lr, ok = compute_loss_recovered(h_with, h_zero, h_orig)
        mse = (h_zero - h_with) / h_zero if h_zero > 0 else 0.0
        old = m.get("loss_recovered")
        parts = path.relative_to(root).parts[:3]
        tag = "/".join(parts)
        if old is None or abs(old - lr) > 0.01 or old < 0:
            print(
                f"{tag}: old={old} new={lr:.4f} valid={ok} "
                f"mse_vs_zero={mse:.3f} "
                f"(H_sae={h_with:.4g} H_zero={h_zero:.4g} H_dense={h_orig:.4g})"
            )


if __name__ == "__main__":
    main()
