# Training-time gradient monitoring

INTERNAL DIAGNOSTIC ONLY

MOSI seed66; paired from-scratch 100-epoch reruns on biggpu GPU7. Original source ad211c0; observer implementation 597e9a1. No architecture/loss changes. Per-rate Test-oracle BEST; one cyclic-missing training trajectory per model, not eight independently trained models.

Verified: 100 gradient-file hashes/model, metric hashes, completed provenance, 200 paired steps with identical masks/counts. Both reproduce the historical seed66 aggregate scores.

| Missing rate | Flat W-F1 (%) | Nested W-F1 (%) | Difference (pp) |
|---|---:|---:|---:|
| 0.0 | 88.205 | 88.078 | -0.127 |
| 0.1 | 86.507 | 86.358 | -0.149 |
| 0.2 | 83.187 | 83.735 | +0.548 |
| 0.3 | 80.763 | 80.523 | -0.240 |
| 0.4 | 80.827 | 81.012 | +0.185 |
| 0.5 | 77.494 | 77.675 | +0.181 |
| 0.6 | 75.790 | 75.032 | -0.758 |
| 0.7 | 75.773 | 75.525 | -0.249 |
| 8-rate mean | 81.068 | 80.992 | -0.076 |
| High missing | 76.352 | 76.077 | -0.275 |

## Full 1–100 epoch input-gradient summary

Sample-count weighted over active observations; batch-mean MSE gradients multiplied by valid utterance count. All epochs included, including zero-initialization. Local includes Skip; Memory uses forward 512 dimensions and excludes first turns/inactive Gap. These are training gradients, not frozen output Jacobians.

| Slot | Flat | Nested | Nested / Flat |
|---|---:|---:|---:|
| Local | 0.768517 | 0.760329 | 0.989 |
| Base | 0.609223 | 0.596014 | 0.978 |
| Gap-A | 0.234871 | 0.233475 | 0.994 |
| Gap-T | 0.293458 | 0.275699 | 0.939 |
| Gap-V | 0.260872 | 0.251414 | 0.964 |

## Optimization and residual checks

- flat: clipping active 200/200 steps; median coefficient 0.057682.
- nested: clipping active 200/200 steps; median coefficient 0.039143.

Nested nonzero residual/input ratios (sample-count weighted):
- Local: 0.011166.
- Base: 0.048911.
- Gap-A: 0.113654.
- Gap-T: 0.159843.
- Gap-V: 0.118900.

## Interpretation limits

Nested receives finite, nonzero parameter gradients and learns nonzero residuals: no evidence of a completely disconnected branch. Smaller aggregate input gradients alone do not establish harmful gradient vanishing. Both models clip every step, so clipping is not a Nested-only failure. Model coordinates, loss, parameter scale and independent joint training confound gradient comparisons. Residual norms do not establish useful sentiment information or conflict detection. Seed66 alone does not establish multi-seed consistency.

gradient_windows.csv provides all eight-epoch windows; 97–100 is explicitly a partial tail, not directly rate-balanced. SUMMARY.json includes parameter-group L2/RMS means. No additional training, inference or automatic multi-seed expansion was performed for this analysis.

Reproduce with: `python experiments/osram_nested_training_gradients_20261010/analyze.py --cache <cached gradients and final JSON files> --output experiments/osram_nested_training_gradients_20261010`.
