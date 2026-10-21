# MOSEI seed66 Direct: frozen learned-gamma versus gamma=1

INTERNAL DIAGNOSTIC ONLY

No training/new mask/weight update/threshold search. One causal Memory scan per batch.
Each rate reuses its existing BEST checkpoint and exact saved mask;within-rate on/off weights and Local are identical.
gamma=1 retains source reads;it is NOT source removal and NOT the separately trained original Flat.

| Scope | Learned ACC | Gamma1 ACC | Learned W-F1 | Gamma1 W-F1 | Delta W-F1(pp) | Corrections | Harms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean | 83.393 | 83.1659 | 83.2436 | 82.883 | 0.3606 | 408 | 342 |
| high | 80.2513 | 79.9853 | 79.9802 | 79.4718 | 0.508427 | 205 | 176 |

## Per-rate results

| Rate | Learned W-F1 | Gamma1 W-F1 | Delta(pp) | Mean abs prediction change | Flips | Corrections | Harms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.1462 | 87.1973 | -0.0511249 | 0.0361952 | 58 | 28 | 30 |
| 0.1 | 86.0631 | 85.6915 | 0.37158 | 0.0507426 | 81 | 47 | 34 |
| 0.2 | 85.4674 | 85.5638 | -0.0963554 | 0.0560717 | 53 | 24 | 29 |
| 0.3 | 84.8698 | 84.8649 | 0.00480298 | 0.0546428 | 77 | 37 | 40 |
| 0.4 | 82.4618 | 81.3312 | 1.13062 | 0.0594872 | 100 | 67 | 33 |
| 0.5 | 81.7857 | 81.4161 | 0.369585 | 0.0767555 | 113 | 60 | 53 |
| 0.6 | 79.5787 | 78.9802 | 0.598454 | 0.0644331 | 125 | 69 | 56 |
| 0.7 | 78.5762 | 78.0189 | 0.557242 | 0.0809776 | 143 | 76 | 67 |

## Text conditions (equal nonempty-rate means)

| Group | Rates | N polarity(exposures) | Learned W-F1 | Gamma1 W-F1 | Delta(pp) | Corrections | Harms |
|---|---:|---:|---:|---:|---:|---:|---:|
| T_visible | 8 | 19059 | 86.7816 | 86.7897 | -0.00811895 | 157 | 157 |
| T_missing_seen | 7 | 7506 | 78.0733 | 76.7115 | 1.36184 | 223 | 146 |
| T_missing_unseen | 7 | 2507 | 66.2325 | 65.5495 | 0.682987 | 28 | 39 |

## Active gamma distributions (pooled rate exposures)

| Slot | N | Mean | Variance | Min | Max | P05 | Median | P95 | Within0.01of1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 172480 | 1.77895 | 0.0056943 | 1.48008 | 1.85948 | 1.61392 | 1.81445 | 1.85784 | 0 |
| A | 57647 | 1.77917 | 0.00565217 | 1.48016 | 1.85948 | 1.61456 | 1.81438 | 1.8579 | 0 |
| T | 57430 | 1.77871 | 0.00574565 | 1.48024 | 1.85945 | 1.61298 | 1.81438 | 1.85772 | 0 |
| V | 57403 | 1.77897 | 0.00568513 | 1.48008 | 1.85947 | 1.61443 | 1.81458 | 1.85796 | 0 |
| Base/A | 29029 | 1.78344 | 0.00554365 | 1.4802 | 1.85948 | 1.61454 | 1.81539 | 1.85894 | 0 |
| Base/T | 28954 | 1.78296 | 0.00564179 | 1.48028 | 1.85941 | 1.61289 | 1.81539 | 1.85877 | 0 |
| Base/V | 28947 | 1.78329 | 0.00557029 | 1.48012 | 1.85947 | 1.6144 | 1.81539 | 1.85894 | 0 |
| Gap-A/A | 9644 | 1.77265 | 0.0057155 | 1.48024 | 1.85948 | 1.6144 | 1.80567 | 1.83617 | 0 |
| Gap-A/T | 9569 | 1.77222 | 0.00578441 | 1.48032 | 1.85945 | 1.61344 | 1.80566 | 1.83616 | 0 |
| Gap-A/V | 9530 | 1.77242 | 0.00574813 | 1.48017 | 1.85947 | 1.61432 | 1.80565 | 1.83612 | 0 |
| Gap-T/A | 9549 | 1.7788 | 0.0056803 | 1.61424 | 1.85948 | 1.61473 | 1.81554 | 1.83555 | 0 |
| Gap-T/T | 9511 | 1.77835 | 0.00577817 | 1.60839 | 1.85939 | 1.6127 | 1.81553 | 1.83552 | 0 |
| Gap-T/V | 9529 | 1.77846 | 0.00571953 | 1.614 | 1.85947 | 1.61465 | 1.81553 | 1.83555 | 0 |
| Gap-V/A | 9425 | 1.77309 | 0.00575642 | 1.48016 | 1.85941 | 1.6144 | 1.80568 | 1.83622 | 0 |
| Gap-V/T | 9396 | 1.77258 | 0.00585704 | 1.48024 | 1.8594 | 1.61336 | 1.80568 | 1.83621 | 0 |
| Gap-V/V | 9397 | 1.77284 | 0.00580108 | 1.48008 | 1.85943 | 1.61424 | 1.80568 | 1.83618 | 0 |

METRICS.csv includes all seven current availability patterns;GAMMA.csv/REPORT.json include per-source/per-role and Text-condition distributions.
Groups are mutually exclusive:currentT visible;currentT missing and strict-pastT seen;currentT missing and strict-pastT unseen.
Zero sentiment labels are excluded only from polarity metrics;MSE includes them. Corrections/harms use gamma1→learned,prediction>0.
Counts across rates are repeated exposures;do not sum group W-F1 or interpret counts as independent unique utterances.
Small effects cannot establish that Flat absorbed calibration,or identify a gradient/parameter-identifiability failure. No mechanism claim or independent-test performance claim.
