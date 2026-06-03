# SAEBench metrics on HiRISE / ViT

[SAEBench](https://github.com/adamkarvonen/SAEBench) targets **language models** (Pythia, Gemma) with `transformer_lens` + text datasets. This project implements the metrics that transfer to **vision patch activations** without installing the full SAEBench stack (~115 min/SAE + LM weights).

## Quick start

```powershell
# Fast: proxies from saved eval artifacts (~seconds)
& "c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe" scripts\compute_saebench.py --proxy-only

# Full SAEBench-aligned Core + HiRISE Sparse Probing (~1–3 min/run, GPU)
# Landform probes on official HiRISE v3.2 test split (311 originals, classes 1–7)
& "c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe" scripts\compute_saebench.py --extract --cache-activations --probe-landforms-only

# Legacy ad-hoc ~87-image holdout (not for paper claims)
# scripts\compute_saebench.py --extract --probe-landforms-only --probe-adhoc-eval

# If CUDA OOM, lower core batch size:
& "c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe" scripts\compute_saebench.py --extract --cache-activations --core-batch-size 512

# Subset
& "c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe" scripts\compute_saebench.py --extract --backbone mars_orbital_vit --cache-activations
```

Progress is saved after **each run** to `saebench_scores.csv`. Failures are logged in `results/saebench_run.log`. With `--cache-activations`, ViT activations are cached in `eval_patch_activations.pt` so retries skip the backbone forward pass.

Outputs: `results/saebench_scores.csv`, per-run `results/.../saebench_metrics.json`

## What maps to SAEBench

| SAEBench eval | In this repo | Notes |
|---------------|--------------|-------|
| **Core** (L0, explained variance, MSE, cossim) | `--extract` → `core_*` columns | Formulas from `sae_bench/evals/core` |
| **Sparse probing** | `--extract --probe-landforms-only` → `sparse_probe_f1_*` | Official creator **test** split (311 landforms, classes 1–7); probe trained on official **train**. Per-class F1 in JSON; `*_core_classes` = macro over classes with ≥20 test images (excludes bright dune, impact ejecta). |
| **Absorption** (Chanin et al.) | `proxy_absorption_fraction` only | Our proxy ≠ first-letter + IG metric; full eval needs LM |
| **AutoInterp** | — | Needs LLM API + text/image captions |
| **RAVEL** | — | Entity-attribute text benchmark |
| **SCR / TPP** | — | Spurious correlation on NLP probes |
| **Unlearning** | — | WMDP-bio / MMLU |

## Installing upstream SAEBench (optional)

Only useful if you also train SAEs on **text LMs**:

```powershell
pip install sae-bench sae-lens transformer-lens
```

Then follow [custom_saes/README](https://github.com/adamkarvonen/SAEBench/tree/main/sae_bench/custom_saes). It will **not** run your Mars ViT checkpoints without a custom activation store.

## Column guide

- `core_explained_variance` — SAEBench corrected variance explained (higher = better reconstruction)
- `core_mean_l0` — average active latents per patch
- `sparse_probe_f1_backbone` — macro-F1 on official test split (311 images)
- `sparse_probe_f1_*_core_classes` — macro-F1 on well-represented test classes only (n≥20)
- `sparse_probe_num_images` — official test images (311)
- `sparse_probe_num_train_images` — official train images used to fit the probe
- `sparse_probe_per_class_f1_*` — per landform breakdown in `saebench_metrics.json` only
- `proxy_absorption_fraction` — shared-latent proxy (lower = less absorption)
