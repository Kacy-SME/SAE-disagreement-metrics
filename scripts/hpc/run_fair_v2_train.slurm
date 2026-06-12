#!/bin/bash
#SBATCH --job-name=sae_fair_v2_train
#SBATCH --output=/home/khatfie2/SAE_Experiments/logs/sae_fair_v2_train_%j.out
#SBATCH --error=/home/khatfie2/SAE_Experiments/logs/sae_fair_v2_train_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=public

set -euo pipefail

REPO="${REPO:-$HOME/SAE_Experiments}"
VENV="${VENV:-$SCRATCH/envs/sae_env}"
POST2025_CACHE="${POST2025_CACHE_DIR:-$SCRATCH/hirise_post2025_cache/patches/extracted}"
MARS_DRIVE="${MARS_DRIVE_ROOT:-$SCRATCH/metrics_models}"

mkdir -p "$HOME/SAE_Experiments/logs"
cd "$REPO"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO"
export POST2025_CACHE_DIR="$POST2025_CACHE"
export MARS_DRIVE_ROOT="$MARS_DRIVE"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_cache}"
export MOMO_CHECKPOINT_PATH="$SCRATCH/metrics_models/momo.pth"
export HIRISE_CTX_THEMIS_CHECKPOINT_PATH="$SCRATCH/metrics_models/hirise_ctx_themis.pth"

BACKBONE_FILTER=(${BACKBONE:-})
LAYER_FILTER=(${LAYER:-})
ARCH_FILTER=(${SAE_ARCH:-})

CMD=(python scripts/sweep_sae_fair_v2.py --protocol winner --train)
if ((${#BACKBONE_FILTER[@]})); then
  for bb in "${BACKBONE_FILTER[@]}"; do CMD+=(--backbone "$bb"); done
fi
if ((${#LAYER_FILTER[@]})); then
  for layer in "${LAYER_FILTER[@]}"; do CMD+=(--layer "$layer"); done
fi
if ((${#ARCH_FILTER[@]})); then
  for arch in "${ARCH_FILTER[@]}"; do CMD+=(--sae-arch "$arch"); done
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
