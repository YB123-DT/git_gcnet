# OSRAM MOSI learning-rate screen

**Internal Test-oracle diagnostic only; not a formal paper result.**

The reference is the existing OSRAM H8/Output700 run at learning rate 1e-3.
The two new variants keep the same backbone, masks, cyclic schedule, features,
Student/Teacher, MMoE, JEPA loss, and optimizer settings; only the learning
rate changes.

## Per-rate Test-oracle comparison

Values are five-seed mean weighted-F1 percentages. Each rate independently
selects its best Test epoch from the recorded 100-epoch history.

| Missing rate | lr=1e-3 | lr=5e-4 | Delta | lr=3e-4 | Delta |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.64±0.67 | 87.68±0.32 | +0.04 | 87.10±0.23 | -0.54 |
| 0.1 | 85.37±0.36 | 85.24±1.03 | -0.12 | 85.11±0.88 | -0.25 |
| 0.2 | 82.69±1.09 | 82.63±1.22 | -0.06 | 82.48±1.00 | -0.21 |
| 0.3 | 81.44±0.65 | 81.16±0.59 | -0.28 | 81.36±1.17 | -0.08 |
| 0.4 | 79.35±1.52 | 79.04±2.24 | -0.31 | 78.90±2.01 | -0.46 |
| 0.5 | 77.30±0.82 | 76.57±1.06 | -0.73 | 76.78±0.95 | -0.52 |
| 0.6 | 76.80±0.53 | 76.56±0.45 | -0.24 | 75.91±0.90 | -0.89 |
| 0.7 | 75.65±1.94 | 74.99±1.55 | -0.66 | 74.95±1.36 | -0.71 |

| Summary | lr=1e-3 | lr=5e-4 | lr=3e-4 |
|---|---:|---:|---:|
| Eight-rate mean | **80.78** | **80.48** | **80.32** |
| High-missing mean (0.5/0.6/0.7) | **76.59** | **76.04** | **75.88** |

The lower learning rates do not improve the model. Relative to 1e-3, 5e-4
loses 0.30 points overall and 0.55 points at high missing rates; 3e-4 loses
0.46 and 0.71 points respectively. The current 1e-3 setting is retained.

## Saved-checkpoint view

The actual training checkpoint uses one eight-rate-mean Test-oracle epoch per
seed. Its five-seed mean is 80.27% for 1e-3, 79.82% for 5e-4, and 79.49% for
3e-4. The per-rate table above is the requested optimistic diagnostic
extraction, not a formal benchmark protocol.

## Decision

**STOP this learning-rate branch.** There is no evidence that lowering the
learning rate fixes the MOSI degradation. Do not expand either lower-rate
variant to MOSEI or IEMOCAP. Further work, if any, must target a separately
justified mechanism rather than repeat this LR screen.
