# MOSI Local Cross-Attention result

**INTERNAL DIAGNOSTIC ONLY: each seed × rate independently selects highest Test W-F1, earliest tie.**
Five seeds66–70, 100epochs, cyclic, causal OSRAM write step .6. Flat inherited from full histories.
Eight-rate/high-rate averages below are descriptive, never used to choose a checkpoint.

| Rate | Flat mean ± SD | Cross-attn mean ± SD | Delta pp | Positive seeds |
|---|---:|---:|---:|---:|
| 0.0 | 87.417 ± 0.224 | 87.261 ± 0.508 | -0.156 | 3/5 |
| 0.1 | 85.398 ± 0.719 | 84.947 ± 0.886 | -0.451 | 1/5 |
| 0.2 | 81.935 ± 1.195 | 81.620 ± 1.280 | -0.315 | 1/5 |
| 0.3 | 80.907 ± 0.890 | 80.576 ± 0.508 | -0.331 | 3/5 |
| 0.4 | 78.088 ± 1.033 | 77.391 ± 1.557 | -0.696 | 1/5 |
| 0.5 | 76.630 ± 1.064 | 75.290 ± 1.002 | -1.339 | 0/5 |
| 0.6 | 75.090 ± 0.835 | 74.864 ± 0.710 | -0.227 | 3/5 |
| 0.7 | 74.553 ± 2.172 | 73.399 ± 2.866 | -1.154 | 0/5 |
| all8 | 80.002 ± 0.468 | 79.419 ± 0.588 | -0.584 | 0/5 |
| high | 75.424 ± 0.877 | 74.518 ± 0.929 | -0.907 | 0/5 |

All test masks matched. Per-seed/rate selected epochs are in per_seed_rate.csv.
This is not a parameter-matched isolated compression experiment. See DESIGN.md.
