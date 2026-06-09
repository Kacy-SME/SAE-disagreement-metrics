#!/bin/bash
#SBATCH --job-name=sae_fair_v2_probe
#SBATCH --output=logs/sae_fair_v2_probe_%j.out
#SBATCH --error=logs/sae_fair_v2_probe_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

REPO="${REPO:-$HOME/SAE-Experiments}"
VENV="${VENV:-$REPO/.venv}"
POST2025_CACHE="${POST2025_CACHE_DIR:-$SCRATCH/hirise_post2025_cache/patches}"
MARS_BENCH="${MARS_BENCH_ROOT:-}"
MARS_DRIVE="${MARS_DRIVE_ROOT:-$SCRATCH/metrics_models}"

mkdir -p logs
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
export POST2025_CACHE_DIR="$POST2025_CACHE"
export MARS_DRIVE_ROOT="$MARS_DRIVE"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"

BACKBONE_FILTER=(${BACKBONE:-})

CMD=(python scripts/sweep_sae_fair_v2.py --protocol winner --probe --plot-histograms)
if [[ -n "$MARS_BENCH" ]]; then
  CMD+=(--marsbench-root "$MARS_BENCH")
fi
if ((${#BACKBONE_FILTER[@]})); then
  for bb in "${BACKBONE_FILTER[@]}"; do CMD+=(--backbone "$bb"); done
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
