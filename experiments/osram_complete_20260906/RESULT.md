# OSRAM result: CMU-MOSI internal diagnostic

**Internal diagnostic only.  Not a formal paper result.**

## Run status

All five OSRAM seeds and all five same-protocol GCNet controls completed
successfully.  There were no NaN/Inf losses, no process failures during the
training runs, and every run evaluated all eight missing rates after every
epoch.

The saved run checkpoint uses the eight-rate-mean Test-oracle rule.  In response
to the later request to inspect the best score for each rate, the table below is
the independently selected per-rate Test-oracle view extracted from the recorded
histories.  Therefore each row/rate can have a different epoch.  This view is
optimistic and is explicitly diagnostic, not a paper protocol.

## Eight-rate per-rate Test-oracle summary

Weighted F1 is shown as a percentage; `delta` is OSRAM minus GCNet.

| Missing rate | GCNet mean ± std | OSRAM mean ± std | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 86.70 ± 0.66 | 87.50 ± 0.25 | +0.80 | 4/5 |
| 0.1 | 84.55 ± 0.57 | 85.01 ± 0.94 | +0.46 | 4/5 |
| 0.2 | 82.14 ± 1.16 | 82.90 ± 0.86 | +0.76 | 4/5 |
| 0.3 | 81.25 ± 1.03 | 81.53 ± 0.34 | +0.28 | 4/5 |
| 0.4 | 78.45 ± 1.51 | 79.45 ± 1.69 | +1.00 | 4/5 |
| 0.5 | 76.55 ± 0.85 | 77.12 ± 1.01 | +0.57 | 3/5 |
| 0.6 | 76.03 ± 0.51 | 76.24 ± 0.59 | +0.20 | 3/5 |
| 0.7 | 74.80 ± 2.07 | 75.48 ± 2.44 | +0.69 | 4/5 |

The overall mean across the eight per-rate means is **80.06% for GCNet** and
**80.65% for OSRAM** (`+0.59` points).  The high-missing mean over 0.5/0.6/0.7
is **75.79% vs 76.28%** (`+0.49` points).

## Seed-level per-rate-oracle means

| Seed | GCNet 8-rate mean | OSRAM 8-rate mean | Delta |
|---:|---:|---:|---:|
| 66 | 80.42 | 80.91 | +0.49 |
| 67 | 80.08 | 80.78 | +0.71 |
| 68 | 79.76 | 80.60 | +0.84 |
| 69 | 80.07 | 80.26 | +0.18 |
| 70 | 79.97 | 80.71 | +0.75 |

All 5/5 seeds are positive in this per-rate diagnostic view.  The saved
single-checkpoint eight-rate-mean Test-oracle runs are slightly lower but remain
positive on average: GCNet **79.56%**, OSRAM **80.05%**, delta **+0.48 points**.

## OSRAM diagnostics

The final-batch diagnostics are stored per seed in `diagnostics.json`.  Across
seeds, the observed-address residual has nonzero activity (`rho` approximately
0.75–0.84 and `eta` approximately 0.27–0.37), and gap context is nonzero.  The
gap/base ratios are approximately 0.36–0.52, so the model did not collapse to
Local + Base only.  Memory alpha stayed near its initialized 0.98 and beta near
0.50.  No prediction or hidden-state non-finiteness was observed.

The high-missing curve still decreases from rate 0.5 to 0.7, as does the
control; OSRAM improves the control modestly rather than removing the inherent
missing-information difficulty.

## Decision

**KEEP for the next planned step, with a narrow claim:** the complete OSRAM
replacement is worth continuing because it improves every per-rate mean, all
five seed-level per-rate means are positive, miss=0 does not regress, and the
gap/residual mechanisms are active.  This is not evidence that OSRAM solves the
high-missing decline, and no ablation or extra tuning was run in this round.

`per_seed_rate.csv` and `summary.csv` contain the machine-readable values;
`raw/` and `raw_control/` retain the source histories and configs used for the
extraction.
