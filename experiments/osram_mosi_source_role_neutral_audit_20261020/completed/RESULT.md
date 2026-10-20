# MOSI 12-slot learned-weight neutralization audit

INTERNAL DIAGNOSTIC ONLY

Frozen inference only. Each condition uses its own already-selected BEST checkpoint and exact mask.
Neutralization is gamma=1, NOT gamma=0 source removal. Local and Memory trajectory stay fixed.
Positive learned-minus-reset W-F1 means that learned slot calibration helps relative to retaining the original source contribution.
The primary cells below are equal eight-rate mean FULL-TEST W-F1 changes (percentage points), not active-subset F1.

## Seed 66: 4 x 3 full-test contribution table

| Read role | Historical A | Historical T | Historical V |
|---|---:|---:|---:|
| Base | +0.040 | +0.020 | +0.000 |
| Gap-A | +0.020 | +0.020 | +0.018 |
| Gap-T | +0.020 | +0.020 | +0.000 |
| Gap-V | +0.000 | +0.000 | +0.037 |

## Seed 67: 4 x 3 full-test contribution table

| Read role | Historical A | Historical T | Historical V |
|---|---:|---:|---:|
| Base | -0.018 | -0.057 | +0.019 |
| Gap-A | +0.000 | +0.000 | +0.000 |
| Gap-T | +0.000 | -0.057 | +0.018 |
| Gap-V | +0.000 | +0.000 | +0.037 |

## Seed 68: 4 x 3 full-test contribution table

| Read role | Historical A | Historical T | Historical V |
|---|---:|---:|---:|
| Base | +0.000 | -0.019 | +0.000 |
| Gap-A | +0.000 | +0.000 | +0.000 |
| Gap-T | +0.000 | +0.001 | +0.000 |
| Gap-V | +0.000 | +0.000 | +0.000 |

## Complete-path comparison

| Seed | Scope | Original Flat W-F1 | All gamma=1 W-F1 | Learned W-F1 | Total pp | Main-path pp | Direct pp |
|---|---|---:|---:|---:|---:|---:|---:|
| 66 | mean | 79.128 | 79.407 | 79.450 | +0.322 | +0.279 | +0.043 |
| 66 | high | 73.338 | 73.630 | 73.478 | +0.140 | +0.292 | -0.152 |
| 67 | mean | 78.751 | 78.615 | 78.578 | -0.172 | -0.136 | -0.036 |
| 67 | high | 72.659 | 72.314 | 72.320 | -0.339 | -0.345 | +0.007 |
| 68 | mean | 78.794 | 78.471 | 78.471 | -0.322 | -0.323 | +0.001 |
| 68 | high | 72.727 | 72.060 | 72.062 | -0.665 | -0.666 | +0.001 |

| Across seeds | Flat W-F1 | Neutral W-F1 | Learned W-F1 | Total pp | Direct pp |
|---|---:|---:|---:|---:|---:|
| mean | 78.891 ± 0.206 | 78.831 ± 0.504 | 78.833 ± 0.536 | -0.058 | +0.002 |
| high | 72.908 ± 0.374 | 72.668 ± 0.842 | 72.620 ± 0.754 | -0.288 | -0.048 |

Total=learned−original Flat; main-path=all gamma1−original Flat; direct=learned−all gamma1.
The differences telescope algebraically, but main-path change is not a causal attribution of training dynamics.
High is rates0.5/0.6/0.7. Means are computed within seed first; ± is sample standard deviation across three seeds.

## Cross-seed directions (descriptive only)

| Slot | Seed66 | Seed67 | Seed68 | Mixed nonzero signs? |
|---|---|---|---|---|
| Base/A | positive | negative | zero | True |
| Base/T | positive | negative | negative | True |
| Base/V | zero | positive | zero | False |
| Gap-A/A | positive | zero | zero | False |
| Gap-A/T | positive | zero | zero | False |
| Gap-A/V | positive | zero | zero | False |
| Gap-T/A | positive | zero | zero | False |
| Gap-T/T | positive | negative | positive | True |
| Gap-T/V | zero | positive | zero | False |
| Gap-V/A | zero | zero | zero | False |
| Gap-V/T | zero | zero | zero | False |
| Gap-V/V | positive | positive | zero | False |

No statistical-significance claim is made. Do not sum the 12 independent nonlinear intervention effects.
Inactive slots have full-test gain0 and active-only N/A, not evidence that a present source is useless.
Corrections/harms count reset→learned polarity changes on the identical nonzero-label active cohort.
N includes neutral labels for MSE; N_polarity excludes label0 and uses prediction>0.
Repeated observations across rates/seeds are not independent utterances; counts must not be interpreted as unique corpus totals.
All per-rate active-cohort results, counts and gamma statistics are in PER_RATE.md and SLOTS.csv.
Gamma magnitude alone is not reliability, semantic conflict, or causal source importance.
