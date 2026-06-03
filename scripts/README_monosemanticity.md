# Monosemanticity scoring (Anthropic 2023 + HiRISE vision)

Paper: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

## What you can run today (no API)

```powershell
& "c:\Users\kacy\Desktop\Orbital_ViT\.venv\Scripts\python.exe" scripts\compute_monosemanticity.py
```

Outputs:

- `results/monosemanticity_scores.csv` — one row per run
- `results/{backbone}/{layer}/{sae}/monosemanticity_proxies.json` — per-run aggregates

### Metrics (vision-adapted proxies)

| Metric | Paper analogue | Interpretation |
|--------|----------------|----------------|
| `mean_label_purity_top20` | Manual / automated activation interpretability | Among top-20 HiRISE images for a latent, how often they share the same semantic label (landform class). Higher → more monosemantic on labels. |
| `mean_interval_label_consistency` | Activation-interval analysis (11 bins) | Whether weaker activations still agree with the top-interval label hypothesis. |
| `median_feature_density` | Feature density histograms | Fraction of eval images where latent fires (>0). |
| `fraction_ultralow_density` | Ultralow-density cluster | Very rare latents (paper treats many as non-interpretable). |
| `absorption_proxy` (from `summary_table.csv`) | Related to SAEBench absorption / polysemanticity | Lower → less feature sharing across images. |

## What the paper uses that we do **not** auto-run

1. **Human rubric (0–14)** — confidence, activation consistency, logit consistency, specificity.  
   For ViT: inspect `feature_activation_matrix.npy` top images per latent (export panels manually).

2. **Automated interpretability — activations** — Claude explains feature from examples, predicts held-out activations; **Spearman correlation** (540 preds/feature).  
   SAEBench uses a similar **detection accuracy** with an LLM judge on text. For HiRISE you would need image captions or a VLM (GPT-4o / Claude vision) and an API key.

3. **Automated interpretability — logit weights** — predict whether tokens are promoted by the feature.  
   **Not applicable** to patch-level ViT features without a hooked classification head.

4. **Loss recovered / MLP NLL fraction** — already in `eval_metrics.json` / `summary_table.csv` as `mse_reduction_vs_zero` and `loss_recovered`.

## Recommended workflow

1. Run `compute_monosemanticity.py` for label-based proxies on all 36 runs.
2. Compare `mean_label_purity_top20` and `absorption_proxy` by backbone/layer (late layer + low absorption ≈ compositional).
3. For paper-faithful **automated** scores, integrate [SAEBench](https://arxiv.org/html/2503.09532) automated interpretability with a VLM on top activating HiRISE crops (future work).
