# SAE Experiments on Mars Orbital ViT Backbones

Structured sparse autoencoder (SAE) experiment pipeline across multiple pretrained ViT backbones and layer depths, using HiRISE v3.2 patch-token activations.

## Paper

This repository accompanies:

> Kacy Hatfield, Lindsay Sanneman. **"When Explanation Metrics Disagree in Planetary AI: Implications for Scientists in the Loop."** SpaceCHI 2026.

Full per-configuration results, dead-latent counts, and effective-dimensionality
values referenced in the paper (Tables 1–3, Figures 1–3, Appendices A–C) are
available under [`results/`](./results).

### Key results

- **Full grid results (36 configurations):** [`results/summary_table.csv`](./results/summary_table.csv)
  — SAE quality (explained variance, cosine similarity, ΔF1) and domain-relevance
  metrics (monosemanticity, probe-decoder alignment, TCAV score/significance)
  for all 6 backbones × 3 layer depths × 2 SAE architectures.
- **Effective dimensionality ($k_{95}$) by backbone and layer:** [`results/k95_by_backbone_layer.csv`](./results/k95_by_backbone_layer.csv)
- **Feature absorption / crater-class directional agreement:** [`results/absorption_crater.csv`](./results/absorption_crater.csv)
- **Dead-latent counts, all 36 configurations:** included in `summary_table.csv` (`n_alive` column)

### Reproducing the paper's numbers
