# Experiment results (summaries only)

Large artifacts (SAE weights, activation caches, heatmaps) stay local and are gitignored.

## Master tables

| File | Description |
|------|-------------|
| `ablations/saebench_fair.csv` | Full fair protocol grid (20 runs, all metrics) |
| `ablations/saebench_fair_paper.csv` | Paper winners (one row per backbone) |
| `ablations/saebench_fair_all.csv` | Best per (backbone, layer, arch) |
| `ablations/saebench_fair_best.csv` | Best per backbone |
| `ablations/momo_density_interventions.csv` | MOMO per_dim vs pca_proj vs zca_whiten |
| `diagnostics/effective_rank.csv` | PCA effective rank across backbones × layers |
| `diagnostics/loss_recovered_cls_zero.csv` | `loss_recovered` with decode(0) vs CLS-zero patch proxy |
| `visualizations/*_top_activations.html` | Self-contained latent inspection reports (base64 thumbnails) |
| `diagnostics/loss_recovered_cls_zero.csv` | CLS-zero vs decode(0) loss_recovered comparison |
| `summary_table.csv` | Legacy `run_experiments.py` proxy metrics |

## Per-run detail

Under `ablations/<backbone>/<tag>/<backbone>/<layer>/<arch>/`:

- `saebench_metrics.json` — SAEBench + interpretability metrics
- `eval_metrics.json` — reconstruction / sparsity proxies from training eval

Weights: `sae_weights.pt` (local only, not in repo).
