#!/bin/bash
#SBATCH --job-name=sae_domars_croma
#SBATCH --output=/home/khatfie2/SAE_Experiments/logs/sae_domars_croma_%j.out
#SBATCH --error=/home/khatfie2/SAE_Experiments/logs/sae_domars_croma_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=public

set -euo pipefail

REPO="${REPO:-$HOME/SAE_Experiments}"
VENV="${VENV:-$SCRATCH/envs/sae_env}"
POST2025_CACHE="${POST2025_CACHE_DIR:-$SCRATCH/hirise_post2025_cache/patches/extracted}"
MARS_BENCH="${MARS_BENCH_ROOT:-}"
MARS_DRIVE="${MARS_DRIVE_ROOT:-$SCRATCH/metrics_models}"

mkdir -p "$HOME/SAE_Experiments/logs"
cd "$REPO"
module load mamba
source activate sae_env
export PYTHONPATH="$REPO"
export POST2025_CACHE_DIR="$POST2025_CACHE"
export MARS_DRIVE_ROOT="$MARS_DRIVE"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_cache}"
export MOMO_CHECKPOINT_PATH="$SCRATCH/metrics_models/momo.pth"
export HIRISE_CTX_THEMIS_CHECKPOINT_PATH="$SCRATCH/metrics_models/hirise_ctx_themis.pth"

CLEAN_RESULTS_DIR="${CLEAN_RESULTS_DIR:-$REPO/results/domars16k_clean}"
mkdir -p "$CLEAN_RESULTS_DIR"

CMD=(python scripts/compute_saebench.py
  --extract
  --results-dir "$REPO/results/ablations"
  --data-dir "$CLEAN_RESULTS_DIR"
  --backbone croma
  --probe-official-test
  --probe-supplementary
  --no-cache-activations
  --marsbench-specs DoMars16k
  --metrics tcav,causal,mono
  --post2025-cache-dir "$POST2025_CACHE"
)
if [[ -n "$MARS_BENCH" ]]; then
  CMD+=(--marsbench-root "$MARS_BENCH")
fi

echo "Backbone: croma"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
