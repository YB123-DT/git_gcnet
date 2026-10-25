# Signature fixed-rate stage-two extension

Source: `c796c48974d4e40c4777264347f3bead0a14b20d`. Internal per-rate Test-oracle exploration.

MOSI66 reuses completed runs; MOSI67/68 train 150 epochs; MOSEI66 trains 60 epochs.
Each seed is averaged over rates first. Reported SD is sample SD across seeds, never rates; single-seed SD is unavailable.

| Dataset | Rate/scope | Seeds | ACC mean ± SD | WF1 mean ± SD | MSE mean ± SD | Δ WF1 mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CMUMOSI | 0.0 | 3 | 87.9573 ± 0.5496 | 87.9229 ± 0.5637 | 0.9889 ± 0.0284 | 0.3858 ± 0.1134 |
| CMUMOSI | 0.1 | 3 | 84.9085 ± 0.4033 | 84.8622 ± 0.4307 | 1.1194 ± 0.0480 | 0.2581 ± 0.3164 |
| CMUMOSI | 0.2 | 3 | 82.8252 ± 0.5353 | 82.7873 ± 0.5989 | 1.3240 ± 0.0443 | 0.0677 ± 0.2147 |
| CMUMOSI | 0.3 | 3 | 81.3516 ± 0.4657 | 81.3653 ± 0.4433 | 1.3249 ± 0.0418 | -0.0227 ± 0.7381 |
| CMUMOSI | 0.4 | 3 | 78.9634 ± 0.8066 | 79.0276 ± 0.8238 | 1.5092 ± 0.0539 | 0.0151 ± 1.1938 |
| CMUMOSI | 0.5 | 3 | 76.9817 ± 1.1906 | 77.0153 ± 1.1222 | 1.5894 ± 0.0378 | 0.8126 ± 0.9666 |
| CMUMOSI | 0.6 | 3 | 74.6443 ± 1.1542 | 74.7655 ± 1.1142 | 1.8246 ± 0.0473 | 1.1171 ± 1.2154 |
| CMUMOSI | 0.7 | 3 | 68.8516 ± 1.5039 | 68.9613 ± 1.5557 | 2.0110 ± 0.0099 | -0.6318 ± 1.5815 |
| CMUMOSI | mean | 3 | 79.5605 ± 0.4677 | 79.5884 ± 0.4707 | 1.4614 ± 0.0203 | 0.2503 ± 0.2421 |
| CMUMOSI | high | 3 | 73.4925 ± 0.4611 | 73.5807 ± 0.4431 | 1.8084 ± 0.0275 | 0.4326 ± 0.1311 |
| CMUMOSEI | 0.0 | 1 | 88.0297 ± NA | 87.9334 ± NA | 0.4545 ± NA | 0.1751 ± NA |
| CMUMOSEI | 0.1 | 1 | 86.9290 ± NA | 86.7987 ± NA | 0.4867 ± NA | 0.1507 ± NA |
| CMUMOSEI | 0.2 | 1 | 86.1860 ± NA | 86.1040 ± NA | 0.5295 ± NA | 0.4064 ± NA |
| CMUMOSEI | 0.3 | 1 | 85.2229 ± NA | 85.0461 ± NA | 0.5585 ± NA | 0.4355 ± NA |
| CMUMOSEI | 0.4 | 1 | 83.5993 ± NA | 83.4246 ± NA | 0.6273 ± NA | 0.0133 ± NA |
| CMUMOSEI | 0.5 | 1 | 82.3886 ± NA | 82.3026 ± NA | 0.6658 ± NA | 0.3416 ± NA |
| CMUMOSEI | 0.6 | 1 | 80.0495 ± NA | 79.9434 ± NA | 0.7233 ± NA | 0.2438 ± NA |
| CMUMOSEI | 0.7 | 1 | 79.1965 ± NA | 78.8245 ± NA | 0.7551 ± NA | 0.7898 ± NA |
| CMUMOSEI | mean | 1 | 83.9502 ± NA | 83.7972 ± NA | 0.6001 ± NA | 0.3195 ± NA |
| CMUMOSEI | high | 1 | 80.5449 ± NA | 80.3568 ± NA | 0.7147 ± NA | 0.4584 ± NA |

| Dataset | Seed | Rate/scope | ACC | WF1 | MSE | Δ WF1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CMUMOSI | 66 | mean | 79.8209 | 79.8539 | 1.4835 | 0.3934 |
| CMUMOSI | 66 | high | 73.8313 | 73.9220 | 1.8320 | 0.5704 |
| CMUMOSI | 67 | mean | 79.8399 | 79.8664 | 1.4437 | 0.3866 |
| CMUMOSI | 67 | high | 73.6789 | 73.7402 | 1.7782 | 0.4180 |
| CMUMOSI | 68 | mean | 79.0206 | 79.0450 | 1.4570 | -0.0292 |
| CMUMOSI | 68 | high | 72.9675 | 73.0799 | 1.8149 | 0.3095 |
| CMUMOSEI | 66 | mean | 83.9502 | 83.7972 | 0.6001 | 0.3195 |
| CMUMOSEI | 66 | high | 80.5449 | 80.3568 | 0.7147 | 0.4584 |
| CMUMOSEI | 66 | 0.0 | 88.0297 | 87.9334 | 0.4545 | 0.1751 |
| CMUMOSEI | 66 | 0.1 | 86.9290 | 86.7987 | 0.4867 | 0.1507 |
| CMUMOSEI | 66 | 0.2 | 86.1860 | 86.1040 | 0.5295 | 0.4064 |
| CMUMOSEI | 66 | 0.3 | 85.2229 | 85.0461 | 0.5585 | 0.4355 |
| CMUMOSEI | 66 | 0.4 | 83.5993 | 83.4246 | 0.6273 | 0.0133 |
| CMUMOSEI | 66 | 0.5 | 82.3886 | 82.3026 | 0.6658 | 0.3416 |
| CMUMOSEI | 66 | 0.6 | 80.0495 | 79.9434 | 0.7233 | 0.2438 |
| CMUMOSEI | 66 | 0.7 | 79.1965 | 78.8245 | 0.7551 | 0.7898 |
| CMUMOSI | 67 | 0.0 | 88.4146 | 88.4029 | 0.9584 | 0.4676 |
| CMUMOSI | 67 | 0.1 | 85.3659 | 85.3510 | 1.0812 | 0.2901 |
| CMUMOSI | 67 | 0.2 | 83.3841 | 83.3717 | 1.2767 | 0.2804 |
| CMUMOSI | 67 | 0.3 | 81.2500 | 81.2158 | 1.3599 | 0.6551 |
| CMUMOSI | 67 | 0.4 | 79.2683 | 79.3694 | 1.5392 | 0.1455 |
| CMUMOSI | 67 | 0.5 | 77.7439 | 77.7924 | 1.5460 | 1.8259 |
| CMUMOSI | 67 | 0.6 | 75.1524 | 75.2956 | 1.7874 | 1.2553 |
| CMUMOSI | 67 | 0.7 | 68.1402 | 68.1324 | 2.0010 | -1.8273 |
| CMUMOSI | 68 | 0.0 | 87.3476 | 87.3022 | 0.9934 | 0.2564 |
| CMUMOSI | 68 | 0.1 | 84.6037 | 84.5388 | 1.1037 | 0.5573 |
| CMUMOSI | 68 | 0.2 | 82.3171 | 82.1749 | 1.3308 | 0.0714 |
| CMUMOSI | 68 | 0.3 | 80.9451 | 81.0161 | 1.3363 | -0.8091 |
| CMUMOSI | 68 | 0.4 | 78.0488 | 78.0879 | 1.4470 | -1.2385 |
| CMUMOSI | 68 | 0.5 | 75.6098 | 75.7287 | 1.6152 | -0.0992 |
| CMUMOSI | 68 | 0.6 | 75.4573 | 75.5155 | 1.8086 | 2.2575 |
| CMUMOSI | 68 | 0.7 | 67.8354 | 67.9956 | 2.0209 | -1.2297 |
| CMUMOSI | 66 | 0.0 | 88.1098 | 88.0634 | 1.0147 | 0.4335 |
| CMUMOSI | 66 | 0.1 | 84.7561 | 84.6967 | 1.1732 | -0.0731 |
| CMUMOSI | 66 | 0.2 | 82.7744 | 82.8152 | 1.3645 | -0.1488 |
| CMUMOSI | 66 | 0.3 | 81.8598 | 81.8641 | 1.2786 | 0.0860 |
| CMUMOSI | 66 | 0.4 | 79.5732 | 79.6254 | 1.5413 | 1.1385 |
| CMUMOSI | 66 | 0.5 | 77.5915 | 77.5248 | 1.6071 | 0.7112 |
| CMUMOSI | 66 | 0.6 | 73.3232 | 73.4852 | 1.8778 | -0.1615 |
| CMUMOSI | 66 | 0.7 | 70.5793 | 70.7560 | 2.0111 | 1.1615 |

## Provenance and limits

Server biggpu, cmerc-repro; frozen Signature stage2 with original GCNet-random-mask-v1. Each run trains and evaluates only its own rate. MOSI seed66 is reused unchanged;67/68 complete150epochs,MOSEI66 completes60epochs. This report does not include the separately running cyclic68 screen. Parent and exact mask hashes/epochs were checked by the finite controller. All32 logical conditions are retained, including seed68's negative overall change. Per-rate Test-oracle, extra stage2 budget, and one MOSEI seed limit interpretation; these are not independent-test generalization claims. MOSEI parent83.477635 is the repaired-RNG LargeFlat, not earlier83.288796.
