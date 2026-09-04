# CMU-MOSEI Text-anchor Residual Test-oracle Diagnostic

## Protocol

- Model: Slot Missing-M3 with `fusion_type=text-anchor-residual`.
- Seeds: 66--70.
- Training: cyclic mixed rates 0.0--0.7, 100 epochs.
- Evaluation: all eight test rates after every epoch.
- Selection: each rate independently selects its highest test W-F1 epoch.

This protocol directly uses the test set for model selection and is therefore
**diagnostic only, not a valid benchmark result**.

Each table cell reports `W-F1 (selected epoch)`.

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Rate mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 87.13 (30) | 86.39 (37) | 85.75 (37) | 84.72 (31) | 84.83 (37) | 83.25 (25) | 82.67 (37) | 81.28 (13) | 84.50 |
| 67 | 86.99 (17) | 86.34 (14) | 85.94 (17) | 84.63 (14) | 83.94 (47) | 82.97 (22) | 82.91 (36) | 82.57 (17) | 84.54 |
| 68 | 87.04 (21) | 86.35 (29) | 85.06 (21) | 84.48 (38) | 83.91 (20) | 82.68 (55) | 82.16 (29) | 81.80 (39) | 84.19 |
| 69 | 87.31 (35) | 86.49 (28) | 85.38 (35) | 84.61 (29) | 83.60 (24) | 82.63 (15) | 82.46 (68) | 80.52 (41) | 84.13 |
| 70 | 86.94 (30) | 85.96 (40) | 84.83 (30) | 84.42 (30) | 83.90 (26) | 83.04 (58) | 81.62 (25) | 81.45 (26) | 84.02 |
| Mean | **87.08** | **86.31** | **85.39** | **84.57** | **84.03** | **82.92** | **82.36** | **81.53** | **84.27** |
| SD | 0.14 | 0.21 | 0.46 | 0.12 | 0.46 | 0.26 | 0.50 | 0.75 | 0.23 |

The result is stable across seeds. Seed-level rate means are 84.50, 84.54,
84.19, 84.13, and 84.02. Miss 0.7 has the largest variance but remains above
80 for every seed.

For context only, the earlier Slot model selected by validation achieved rate
means 86.43, 85.71, 84.66, 83.98, 83.14, 82.28, 81.42, and 80.65 (overall
83.53). The new Test-oracle mean is 84.27 (+0.74), but this is not a valid
paired comparison because both the selection rule and fusion module differ.
A five-seed Slot Test-oracle control or a validation-selected text-anchor run
would be required to isolate the module effect.

## Integrity

- Five histories contain exactly 100 epochs each.
- All 4,000 rate-by-epoch W-F1 values are finite.
- Forty prediction NPZ files were retained; checkpoints were excluded.
- Per-rate maxima are computed from histories. Final NPZ files correspond to
  the trainer's single mean-oracle checkpoint, not all independent epochs.
