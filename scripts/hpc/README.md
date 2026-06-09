# HPC: v2 fair SAE sweep

Train SAEs on post-2025 HiRISE patch cache; evaluate on Mars-Bench.

## 1. Clone and environment

```bash
git clone https://github.com/Kacy-SME/SAE_Experiments.git ~/SAE-Experiments
cd ~/SAE-Experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data on the cluster

**Post-2025 patch cache** (required for training):

```bash
# From your workstation (example):
rsync -avP D:/hirise_post2025_cache/patches/ user@hpc:$SCRATCH/hirise_post2025_cache/patches/
```

Set `POST2025_CACHE_DIR` to that path (SLURM scripts default to `$SCRATCH/hirise_post2025_cache/patches`).

**Mars-Bench** (required for probe/eval): either rsync a local copy to `$SCRATCH/Mars-Bench` and set `MARS_BENCH_ROOT`, or let the loader download from Hugging Face into `$HF_HOME`.

**Backbone checkpoints** (MOMO, hirise_ctx_themis, etc.): set `MARS_DRIVE_ROOT` to a directory containing your checkpoint tree, or set per-model env vars (`MOMO_CHECKPOINT_PATH`, `HIRISE_CTX_THEMIS_CHECKPOINT_PATH`).

## 3. Submit jobs

```bash
mkdir -p logs

# Full v2 winner train (24 runs, ~5–6 h each on A100-class GPU)
sbatch scripts/hpc/run_fair_v2_train.slurm

# Single backbone smoke on login node
python scripts/sweep_sae_fair_v2.py --protocol winner --train --smoke --limit 1 \
  --backbone momo --layer middle --sae-arch topk

# After all trains finish — probe + v2 CSV tables
sbatch scripts/hpc/run_fair_v2_probe.slurm
```

Filter one backbone/layer/arch via env vars:

```bash
BACKBONE=momo LAYER=middle SAE_ARCH=topk sbatch scripts/hpc/run_fair_v2_train.slurm
```

## 4. Outputs

| Artifact | Path |
|----------|------|
| Per-run weights | `results/ablations/<backbone>/<tag>/.../sae_weights.pt` (gitignored) |
| Fair v2 table | `results/ablations/saebench_fair_v2.csv` |
| Paper picks | `results/ablations/saebench_fair_paper_v2.csv` |
| Histograms | `figures/metric_histograms/` |

v1 CSVs (`saebench_fair_paper.csv`, etc.) are unchanged by the v2 sweep.
