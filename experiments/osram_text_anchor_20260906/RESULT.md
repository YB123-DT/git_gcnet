# OSRAM + Text-anchor residual result

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**

The experiment changes only `fusion_type` from `mean` to
`text-anchor-residual` on the OSRAM + Missing-M3 configuration. The mask bank,
features, cyclic schedule, fold, seeds, and all training losses are paired
with the existing OSRAM + mean control.

## One checkpoint per seed (8-rate mean Test-oracle selection)

| seed | control epoch / text-anchor epoch | OSRAM + mean | OSRAM + text-anchor | delta |
|---:|---:|---:|---:|---:|
| 66 | 55 vs 63 | 80.4922% | 79.9871% | -0.5051 pp |
| 67 | 49 vs 70 | 80.0066% | 79.1075% | -0.8991 pp |
| 68 | 38 vs 45 | 80.1433% | 79.4432% | -0.7001 pp |
| 69 | 36 vs 38 | 79.6087% | 80.0804% | +0.4717 pp |
| 70 | 49 vs 49 | 79.9904% | 80.0069% | +0.0165 pp |
| **mean** |  | **80.0482%** | **79.7250%** | **-0.3232 pp** |

High-missing (0.5/0.6/0.7) mean under the same one-checkpoint-per-seed rule
is 75.8004% for OSRAM + mean and 75.0126% for text-anchor (-0.7879 pp).

## Per-rate Test-oracle diagnostic

This secondary table takes the best Test epoch independently for each
seed/rate. It is intentionally optimistic and is not the main selection rule.

| rate | OSRAM + mean | Text-anchor | delta | text-anchor positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 87.4998% | 86.9273% | -0.5724 pp | 0/5 |
| 0.1 | 85.0062% | 84.7127% | -0.2936 pp | 3/5 |
| 0.2 | 82.9030% | 82.1526% | -0.7505 pp | 1/5 |
| 0.3 | 81.5311% | 81.6254% | +0.0943 pp | 3/5 |
| 0.4 | 79.4524% | 79.0918% | -0.3607 pp | 1/5 |
| 0.5 | 77.1163% | 76.3437% | -0.7725 pp | 1/5 |
| 0.6 | 76.2353% | 76.3959% | +0.1606 pp | 3/5 |
| 0.7 | 75.4846% | 74.7967% | -0.6879 pp | 2/5 |
| **8-rate mean** | **80.6536%** | **80.2558%** | **-0.3978 pp** |  |

The high-missing per-rate-oracle mean is 76.2787% for OSRAM + mean and
75.8454% for text-anchor (-0.4333 pp). Text-anchor is higher at only two of
the eight rate means (0.3 and 0.6), and it is lower at the complete rate 0.0.

## Diagnostics

The OSRAM address residual remains active and the gap context is nonzero. At
the last recorded batch, the five seeds have gap/base ratios in approximately
0.40–0.58 and address-residual `rho` values approximately 0.78–0.81 (see
`diagnostics.json`). There is no evidence that the text-anchor run collapsed
or that the residual operator was bypassed.

## Decision

**STOP this text-anchor variant as a primary improvement for MOSI.** Under the
paired Test-oracle diagnostics it lowers the 8-rate mean (both one-checkpoint
and per-rate-oracle summaries), and the high-missing mean also decreases.
The negative result does not invalidate OSRAM + mean; it rejects adding this
Text-anchor residual on top of the current OSRAM configuration.
