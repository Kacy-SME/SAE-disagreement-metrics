# SAE Experiments on Mars Orbital ViT Backbones

Structured sparse autoencoder (SAE) experiment pipeline across multiple pretrained ViT backbones and layer depths, using HiRISE v3.2 patch-token activations.

## Paper

This repository accompanies:

> Kacy Hatfield, Lindsay Sanneman. **"When Explanation Metrics Disagree in Planetary AI: Implications for Scientists in the Loop."** SpaceCHI 2026.

Full per-configuration results, dead-latent counts, and effective-dimensionality
values referenced in the paper (Tables 1–3, Figures 1–3, Appendices A–C) are
available under [`results/`](./results).

### Key results

**Monosemanticity vs. TCAV score, anti-correlated across the 36-configuration grid:**

![Monosemanticity vs TCAV](./results/figures/figure1_mono_tcav_scatter.jpg)

**The crater case: the pipeline-selected latent does not resemble craters:**

![Crater case](./results/figures/figure2_crater_case.jpg)

**Effective dimensionality ($k_{95}$) by backbone and layer depth:**

![k95 by layer](./results/figures/figure3_k95_by_layer.jpg)

Full per-configuration CSV results (all 36 grid cells): [`results/summary_table.csv`](./results/summary_table.csv)

