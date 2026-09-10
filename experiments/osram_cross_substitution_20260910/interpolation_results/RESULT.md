# Fixed-sum Base/Gap interpolation

**INTERNAL DIAGNOSTIC ONLY.** Frozen MOSI Flat causal eta=.6, seeds66–70.
No training or new epoch selection. Existing historical eight-rate-mean Test-selected best.pt checkpoints.
Local/memory unchanged; only the Base and unique active Gap emotion slots are redistributed.
U=(B+G)/2, D=(B-G)/2; B(t)=U+tD, G(t)=U-tD. Endpoints copy original tensors exactly.
Evidence sum conserved within FP32 tolerance (1e-6 absolute / 1e-5 relative).
Scores are raw regression predictions; sign threshold zero; nonzero-label AT/AV/TV samples only.
All means: equal rates .1–.7 within seed, then equal seeds. Rate0 has no eligible samples.
SD is across five within-seed rate means. Subset W-F1 is not the full-test benchmark score.

## ALL

| t | W-F1 mean ± SD (%) | Score MAE vs t=1 | P90 abs deviation (group-averaged) | Sign flips vs t=1 |
|---|---:|---:|---:|---:|
| -1.0 | 79.206 ± 0.874 | 0.04233 | 0.10839 | 0.980% |
| -0.5 | 79.243 ± 0.811 | 0.03122 | 0.07789 | 0.803% |
| 0.0 | 79.273 ± 0.859 | 0.02047 | 0.04999 | 0.622% |
| 0.5 | 79.128 ± 0.998 | 0.01008 | 0.02378 | 0.309% |
| 1.0 | 79.182 ± 1.094 | 0.00000 | 0.00000 | 0.000% |

| Component | Signed mean | Mean abs | P90 abs (group-averaged) |
|---|---:|---:|---:|
| odd1 | -0.010099 | 0.021163 | 0.054194 |
| even1 | +0.002555 | 0.004840 | 0.011405 |
| oddhalf | -0.005117 | 0.010719 | 0.027539 |
| evenhalf | +0.000672 | 0.001249 | 0.002970 |

## AT

| t | W-F1 mean ± SD (%) | Score MAE vs t=1 | P90 abs deviation (group-averaged) | Sign flips vs t=1 |
|---|---:|---:|---:|---:|
| -1.0 | 86.974 ± 1.670 | 0.03530 | 0.07161 | 0.325% |
| -0.5 | 87.122 ± 1.670 | 0.02599 | 0.05427 | 0.174% |
| 0.0 | 87.119 ± 1.595 | 0.01734 | 0.03610 | 0.177% |
| 0.5 | 87.007 ± 1.522 | 0.00875 | 0.01831 | 0.067% |
| 1.0 | 87.073 ± 1.614 | 0.00000 | 0.00000 | 0.000% |

| Component | Signed mean | Mean abs | P90 abs (group-averaged) |
|---|---:|---:|---:|
| odd1 | -0.006550 | 0.017650 | 0.035804 |
| even1 | +0.002725 | 0.004633 | 0.010858 |
| oddhalf | -0.003306 | 0.008964 | 0.018250 |
| evenhalf | +0.000709 | 0.001188 | 0.002799 |

## AV

| t | W-F1 mean ± SD (%) | Score MAE vs t=1 | P90 abs deviation (group-averaged) | Sign flips vs t=1 |
|---|---:|---:|---:|---:|
| -1.0 | 63.265 ± 2.787 | 0.02407 | 0.04518 | 1.965% |
| -0.5 | 63.326 ± 2.514 | 0.02009 | 0.03700 | 1.644% |
| 0.0 | 63.293 ± 2.657 | 0.01473 | 0.02691 | 1.202% |
| 0.5 | 62.971 ± 3.021 | 0.00801 | 0.01473 | 0.544% |
| 1.0 | 62.959 ± 3.093 | 0.00000 | 0.00000 | 0.000% |

| Component | Signed mean | Mean abs | P90 abs (group-averaged) |
|---|---:|---:|---:|
| odd1 | +0.007674 | 0.012036 | 0.022590 |
| even1 | +0.001551 | 0.003598 | 0.007026 |
| oddhalf | +0.003876 | 0.006086 | 0.011429 |
| evenhalf | +0.000396 | 0.000911 | 0.001780 |

## TV

| t | W-F1 mean ± SD (%) | Score MAE vs t=1 | P90 abs deviation (group-averaged) | Sign flips vs t=1 |
|---|---:|---:|---:|---:|
| -1.0 | 86.840 ± 1.992 | 0.06763 | 0.12059 | 0.685% |
| -0.5 | 86.760 ± 2.027 | 0.04749 | 0.08369 | 0.610% |
| 0.0 | 86.861 ± 1.973 | 0.02922 | 0.05212 | 0.508% |
| 0.5 | 86.787 ± 1.956 | 0.01341 | 0.02425 | 0.310% |
| 1.0 | 86.848 ± 1.914 | 0.00000 | 0.00000 | 0.000% |

| Component | Signed mean | Mean abs | P90 abs (group-averaged) |
|---|---:|---:|---:|
| odd1 | -0.031135 | 0.033816 | 0.060293 |
| even1 | +0.003410 | 0.006340 | 0.014525 |
| oddhalf | -0.015780 | 0.017113 | 0.030583 |
| evenhalf | +0.000917 | 0.001661 | 0.003836 |

O(t)=(f(t)-f(-t))/2; E(t)=(f(t)+f(-t))/2-f(0). Applied per sample BEFORE aggregation.
Odd/even here are output sensitivities along the chosen redistribution path, not proven internal modules.
Constant W-F1 does not imply unchanged scores or individual predictions. Endpoint parity alone does not establish exchange symmetry.
Small even sensitivity does not identify why a separately retrained gated architecture failed; bottleneck, projections and optimization remain unseparated explanations.
No prespecified equivalence threshold or population significance claim. Inspect pattern/rate/seed tables and distributions.
