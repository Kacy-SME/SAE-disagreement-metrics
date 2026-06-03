# SAE Experiments on Mars Orbital ViT Backbones

Structured sparse autoencoder (SAE) experiment pipeline across multiple pretrained ViT backbones and layer depths, using HiRISE v3.2 patch-token activations.

## Backbones (6)

| Name | Source |
|------|--------|
| `momo` | MOMO ViT-Base (Mirali33/MOMO or local checkpoint) |
| `dinov3_sat493m` | DINOv3 ViT-L SAT-493M (timm) |
| `mars_orbital_vit` | Custom DINOv2-style Mars ViT (`best_backbone.pt`) |
| `prithvi_eo_2` | Prithvi-EO-2.0-300M (HuggingFace) |
| `croma` | CROMA optical encoder (antofuller/CROMA) |
| `satmae_pp` | SatMAE++ ViT-L FMoW-RGB (mubashir04) |

Layer depths: `early` (L/4), `middle` (L/2), `late` (3L/4).

SAE architectures: `topk` (k=40, dict=4×d), `matryoshka` (nested d, 2d, 4d; k=40).

**Total runs:** 6 × 3 × 2 = **36** combinations.

## Data paths (no separate test folder)

You only need **one** HiRISE location:

| Variable / flag | Purpose |
|-----------------|--------|
| `HIRISE_IMAGES_DIR` / `--hirise_images_dir` | Folder with real `map-proj-v3_2` JPGs |
| `HIRISE_LABELS_FILE` / `--hirise_labels_file` | `labels-map-proj-v3_2.txt` |

The pipeline splits that set **80% train / 20% eval** (seeded, reproducible). Eval is not a second dataset path — it is held-out images from the same files. The exact file lists are saved to `results/data_split_manifest.json`.

**No synthetic data:** corrupt images are **skipped** (logged in `results/skipped_hirise_images.json`), not replaced with black frames. Use `--strict-data` to fail instead of skip. A one-time scan caches readable paths in `results/hirise_readable_cache.json` so runs do not crash mid-extraction. Backbone weights must load from real checkpoints (no random Prithvi fallback). Prithvi expects 6 bands; HiRISE is RGB — Prithvi runs need `--allow-prithvi-rgb-proxy` if you include that backbone.

## Setup

```powershell
cd SAE-Experiments
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Set paths (defaults assume Google Drive on Windows):

```powershell
$env:MARS_DRIVE_ROOT = "G:\My Drive\metrics&models"
$env:HIRISE_IMAGES_DIR = "$env:MARS_DRIVE_ROOT\mars_data\hirise_v3_2\images"
$env:HIRISE_LABELS_FILE = "$env:MARS_DRIVE_ROOT\mars_data\hirise_v3_2\labels-map-proj-v3_2.txt"
$env:MOMO_CHECKPOINT_PATH = "path\to\local\momo.pth"  # optional
```

## Progress bars

While running, you should see tqdm bars for:

- Overall experiment loop (`Overall progress`)
- Activation extraction (`Extract train/eval activations`)
- SAE training steps (`SAE training`)
- Buffer refills (`Refill activation buffer`)
- Evaluation (`Eval SAE metrics`, `Feature activation matrix`)

## Usage

Preview the experiment grid:

```powershell
python run_experiments.py --dry_run
```

Run all experiments (resumable — skips existing `sae_weights.pt`):

```powershell
python run_experiments.py
```

Filter runs:

```powershell
python run_experiments.py --backbone mars_orbital_vit --layer middle --sae_arch topk
python run_experiments.py --force  # re-run even if outputs exist
```

## Outputs

Per `(backbone, layer_depth, sae_arch)`:

```
results/{backbone}/{layer_depth}/{sae_arch}/
  sae_weights.pt
  training_loss_curve.npy      # [steps, 2] recon + aux loss
  activation_stats.json
  feature_activation_matrix.npy
  eval_metrics.json
```

Summary: `results/summary_table.csv`

## RTX 3070 Notes

- Activation extraction uses batch_size=16 (configurable in `configs/experiment.yaml`).
- Train activations are **streamed** into a 500k-vector buffer (not held all in RAM).
- DINOv3 ViT-L and SatMAE++ ViT-L are large; run one backbone at a time if VRAM is tight.
- Reduce `training.batch_size` if SAE training CUDA OOMs (default 1024).
- Crashes are appended to `results/run.log`.

Quick smoke test (~10–20 min on RTX 3070):

```powershell
python run_experiments.py --backbone mars_orbital_vit --layer middle --sae_arch topk --smoke
```

`--max-train-images` alone still runs **50k** SAE steps — use `--smoke` for a true sanity check.

## Config

- `configs/backbones.yaml` — backbone registry (model IDs, layers, patch size, preprocessing)
- `configs/experiment.yaml` — training hyperparameters, data split, buffer size
