#!/usr/bin/env python3
"""
Run fair-protocol winner sweep for v2 backbones (original 5 + hirise_ctx_themis).

Writes to saebench_fair_v2.csv / saebench_fair_paper_v2.csv without touching v1 tables.

Examples:
  python scripts/sweep_sae_fair_v2.py --protocol winner --train --probe
  python scripts/sweep_sae_fair_v2.py --protocol winner --probe --backbone hirise_ctx_themis
  python scripts/sweep_sae_fair_v2.py --rebuild-tables
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP = PROJECT_ROOT / "scripts" / "sweep_sae_fair.py"
PYTHON = Path(sys.executable)


def main() -> None:
    cmd = [str(PYTHON), str(SWEEP), "--fair-version", "v2", *sys.argv[1:]]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
