# MOSI Second GraphConv Ablation — Test-oracle Diagnostic

## Comparison

- Control: Text-anchor Missing-M3, `RGCNConv -> GraphConv -> Post-BiLSTM`.
- Treatment: identical model with `RGCNConv -> Post-BiLSTM` in both graph branches.
- Five seeds, 100 epochs, cyclic mixed-rate training.
- Every missing rate independently selects its best Test W-F1 epoch.

The control histories are inherited from commit `62208ae`. This selection rule
directly uses Test and is diagnostic rather than a valid benchmark protocol.

## Treatment scores

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 86.80 | 85.66 | 82.35 | 81.27 | 78.89 | 76.51 | 75.23 | 74.57 | 80.16 |
| 67 | 85.82 | 83.69 | 82.03 | 79.76 | 75.98 | 76.61 | 77.06 | 75.03 | 79.50 |
| 68 | 86.46 | 84.62 | 82.27 | 80.74 | 81.11 | 77.06 | 77.17 | 70.53 | 79.99 |
| 69 | 86.20 | 84.01 | 81.79 | 81.92 | 77.99 | 77.07 | 75.97 | 77.52 | 80.31 |
| 70 | 86.67 | 84.91 | 81.21 | 80.91 | 81.49 | 76.59 | 76.86 | 75.22 | 80.48 |
| Mean | **86.39** | **84.58** | **81.93** | **80.92** | **79.09** | **76.77** | **76.46** | **74.57** | **80.09** |

## Paired result

| Miss | Identity | GraphConv control | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 86.39 | 86.71 | -0.32 | 2/5 |
| 0.1 | 84.58 | 85.20 | -0.62 | 1/5 |
| 0.2 | 81.93 | 82.41 | -0.48 | 1/5 |
| 0.3 | 80.92 | 81.08 | -0.16 | 4/5 |
| 0.4 | 79.09 | 79.09 | +0.00 | 2/5 |
| 0.5 | 76.77 | 76.92 | -0.15 | 2/5 |
| 0.6 | 76.46 | 76.39 | +0.07 | 3/5 |
| 0.7 | 74.57 | 74.82 | -0.25 | 1/5 |
| Mean | **80.09** | **80.33** | **-0.24** | -- |

Seed-level mean deltas are `-0.60`, `-1.11`, `+0.28`, `+0.19`, and `+0.04`.
Removing the second GraphConv helps three of five seeds in the overall mean,
but losses on seeds 66 and 67 dominate. Only miss 0.6 shows a non-negligible
positive mean, and it is just `+0.07` point. The second GraphConv is therefore
not the source of the MOSI ceiling and should remain enabled.

## Integrity

- Five histories contain exactly 100 contiguous epochs.
- All 4,000 rate-by-epoch scores are finite.
- Forty prediction NPZ files are retained and checkpoints are excluded.
- `graph_second_layer=identity` is recorded in every treatment config.
