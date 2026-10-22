# Same-history current-pattern and history-dependence audit

INTERNAL DIAGNOSTIC ONLY

Frozen MOSI seed66 CHM; no training or optimizer updates. One original checkpoint, not per-rate BEST scores.
The source checkpoint was originally selected with Test-oracle; these are exploratory model-behavior results.

## Experiment 1: historical-write gradient geometry

| Group | Pairs | Defined loss cosines | Mean loss cosine | Negative fraction | Mean prediction-Jacobian cosine |
|---|---:|---:|---:|---:|---:|
| Writer | 1092 | 1092 | 0.3764 | 0.2225 | 0.6405 |
| WriteKey | 1092 | 1092 | 0.3364 | 0.2280 | 0.5785 |
| WriteValue | 1092 | 1092 | 0.3875 | 0.2216 | 0.6632 |

For scalar MSE, grad(loss)=2*(prediction-label)*grad(prediction). Opposite residual signs can invert loss-gradient cosine even with aligned prediction Jacobians. Zero-norm cosine is undefined, not zero.
Negative cosine alone does not establish a CHM defect or harmful optimization. These are paired hypothetical current-pattern losses on the same frozen history, not seven independent optimizer updates; ordinary BPTT already reaches historical writes.

Only historical K/V graph nodes are credited here; parameter-sharing paths through current observations are excluded. The collection implementation and STATUS provenance determine the precise graph intervention.

### Writer loss_cosine

| Pattern | A | T | V | AT | AV | TV | ATV |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | N/A | 0.1223 | 0.4093 | 0.3546 | 0.5313 | 0.2551 | 0.2765 |
| T | 0.1223 | N/A | 0.1153 | 0.6140 | 0.1198 | 0.4331 | 0.3915 |
| V | 0.4093 | 0.1153 | N/A | 0.2140 | 0.7008 | 0.3953 | 0.3366 |
| AT | 0.3546 | 0.6140 | 0.2140 | N/A | 0.2307 | 0.4390 | 0.5833 |
| AV | 0.5313 | 0.1198 | 0.7008 | 0.2307 | N/A | 0.3504 | 0.3342 |
| TV | 0.2551 | 0.4331 | 0.3953 | 0.4390 | 0.3504 | N/A | 0.6969 |
| ATV | 0.2765 | 0.3915 | 0.3366 | 0.5833 | 0.3342 | 0.6969 | N/A |

### Writer jacobian_cosine

| Pattern | A | T | V | AT | AV | TV | ATV |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | N/A | 0.5668 | 0.6167 | 0.7186 | 0.7185 | 0.5104 | 0.5099 |
| T | 0.5668 | N/A | 0.4370 | 0.8421 | 0.4147 | 0.6701 | 0.5752 |
| V | 0.6167 | 0.4370 | N/A | 0.5488 | 0.8252 | 0.7166 | 0.6479 |
| AT | 0.7186 | 0.8421 | 0.5488 | N/A | 0.5678 | 0.7118 | 0.7124 |
| AV | 0.7185 | 0.4147 | 0.8252 | 0.5678 | N/A | 0.6311 | 0.6462 |
| TV | 0.5104 | 0.6701 | 0.7166 | 0.7118 | 0.6311 | N/A | 0.8618 |
| ATV | 0.5099 | 0.5752 | 0.6479 | 0.7124 | 0.6462 | 0.8618 | N/A |

## Experiment 2: history dependence

MSE includes neutral labels; polarity ACC/W-F1 exclude y=0 and use prediction>0. All conditions keep current input fixed. Replacements without a valid donor are unavailable, not zero-valued predictions.

### all

| Pattern | History | N | Polarity N | ACC (%) | W-F1 (%) | MSE |
|---|---|---:|---:|---:|---:|---:|
| A | correct | 686 | 656 | 63.720 | 60.577 | 2.3547 |
| A | empty | 686 | 656 | 51.829 | 52.016 | 2.7116 |
| A | replaced | 686 | 656 | 51.372 | 50.284 | 3.1978 |
| T | correct | 686 | 656 | 86.280 | 86.244 | 0.9024 |
| T | empty | 686 | 656 | 86.280 | 86.274 | 0.9961 |
| T | replaced | 686 | 656 | 86.128 | 86.061 | 1.0284 |
| V | correct | 686 | 656 | 63.262 | 63.486 | 2.3753 |
| V | empty | 686 | 656 | 56.555 | 55.496 | 2.7256 |
| V | replaced | 686 | 656 | 58.232 | 58.444 | 2.5527 |
| AT | correct | 686 | 656 | 86.433 | 86.308 | 0.8810 |
| AT | empty | 686 | 656 | 86.433 | 86.393 | 0.9499 |
| AT | replaced | 686 | 656 | 85.671 | 85.510 | 0.9900 |
| AV | correct | 686 | 656 | 62.500 | 62.580 | 2.3709 |
| AV | empty | 686 | 656 | 49.543 | 43.423 | 2.9332 |
| AV | replaced | 686 | 656 | 52.896 | 53.023 | 2.6918 |
| TV | correct | 686 | 656 | 86.433 | 86.423 | 0.9676 |
| TV | empty | 686 | 656 | 84.909 | 84.973 | 1.0780 |
| TV | replaced | 686 | 656 | 86.738 | 86.714 | 1.0017 |
| ATV | correct | 686 | 656 | 86.738 | 86.605 | 0.9368 |
| ATV | empty | 686 | 656 | 86.890 | 86.847 | 0.9681 |
| ATV | replaced | 686 | 656 | 86.890 | 86.754 | 0.9651 |

| Pattern | Control → Real history | Paired N | ΔMSE (control−real) | ΔACC (pp) | ΔW-F1 (pp) | Corrections | Harms |
|---|---|---:|---:|---:|---:|---:|---:|
| A | empty → correct | 686 | 0.3569 | 11.890 | 8.561 | 167 | 89 |
| A | replaced → correct | 686 | 0.8432 | 12.348 | 10.293 | 171 | 90 |
| T | empty → correct | 686 | 0.0937 | 0.000 | -0.030 | 6 | 6 |
| T | replaced → correct | 686 | 0.1260 | 0.152 | 0.183 | 15 | 14 |
| V | empty → correct | 686 | 0.3502 | 6.707 | 7.990 | 80 | 36 |
| V | replaced → correct | 686 | 0.1773 | 5.030 | 5.042 | 75 | 42 |
| AT | empty → correct | 686 | 0.0688 | 0.000 | -0.084 | 9 | 9 |
| AT | replaced → correct | 686 | 0.1090 | 0.762 | 0.798 | 13 | 8 |
| AV | empty → correct | 686 | 0.5623 | 12.957 | 19.157 | 142 | 57 |
| AV | replaced → correct | 686 | 0.3209 | 9.604 | 9.557 | 143 | 80 |
| TV | empty → correct | 686 | 0.1104 | 1.524 | 1.450 | 20 | 10 |
| TV | replaced → correct | 686 | 0.0341 | -0.305 | -0.291 | 7 | 9 |
| ATV | empty → correct | 686 | 0.0313 | -0.152 | -0.242 | 10 | 11 |
| ATV | replaced → correct | 686 | 0.0283 | -0.152 | -0.149 | 7 | 8 |

### history_present

| Pattern | History | N | Polarity N | ACC (%) | W-F1 (%) | MSE |
|---|---|---:|---:|---:|---:|---:|
| A | correct | 655 | 630 | 64.127 | 60.815 | 2.3181 |
| A | empty | 655 | 630 | 51.746 | 51.989 | 2.6919 |
| A | replaced | 655 | 630 | 51.270 | 50.165 | 3.2012 |
| T | correct | 655 | 630 | 86.508 | 86.472 | 0.8647 |
| T | empty | 655 | 630 | 86.508 | 86.504 | 0.9629 |
| T | replaced | 655 | 630 | 86.349 | 86.279 | 0.9967 |
| V | correct | 655 | 630 | 63.175 | 63.435 | 2.3489 |
| V | empty | 655 | 630 | 56.190 | 55.181 | 2.7157 |
| V | replaced | 655 | 630 | 57.937 | 58.166 | 2.5346 |
| AT | correct | 655 | 630 | 86.667 | 86.535 | 0.8473 |
| AT | empty | 655 | 630 | 86.667 | 86.626 | 0.9194 |
| AT | replaced | 655 | 630 | 85.873 | 85.701 | 0.9614 |
| AV | correct | 655 | 630 | 62.222 | 62.398 | 2.3362 |
| AV | empty | 655 | 630 | 48.730 | 42.660 | 2.9251 |
| AV | replaced | 655 | 630 | 52.222 | 52.466 | 2.6723 |
| TV | correct | 655 | 630 | 86.667 | 86.659 | 0.9357 |
| TV | empty | 655 | 630 | 85.079 | 85.155 | 1.0513 |
| TV | replaced | 655 | 630 | 86.984 | 86.961 | 0.9713 |
| ATV | correct | 655 | 630 | 86.984 | 86.844 | 0.9082 |
| ATV | empty | 655 | 630 | 87.143 | 87.100 | 0.9410 |
| ATV | replaced | 655 | 630 | 87.143 | 86.998 | 0.9378 |

| Pattern | Control → Real history | Paired N | ΔMSE (control−real) | ΔACC (pp) | ΔW-F1 (pp) | Corrections | Harms |
|---|---|---:|---:|---:|---:|---:|---:|
| A | empty → correct | 655 | 0.3738 | 12.381 | 8.825 | 167 | 89 |
| A | replaced → correct | 655 | 0.8831 | 12.857 | 10.650 | 171 | 90 |
| T | empty → correct | 655 | 0.0981 | 0.000 | -0.033 | 6 | 6 |
| T | replaced → correct | 655 | 0.1320 | 0.159 | 0.192 | 15 | 14 |
| V | empty → correct | 655 | 0.3668 | 6.984 | 8.254 | 80 | 36 |
| V | replaced → correct | 655 | 0.1857 | 5.238 | 5.269 | 75 | 42 |
| AT | empty → correct | 655 | 0.0721 | 0.000 | -0.092 | 9 | 9 |
| AT | replaced → correct | 655 | 0.1141 | 0.794 | 0.833 | 13 | 8 |
| AV | empty → correct | 655 | 0.5889 | 13.492 | 19.738 | 142 | 57 |
| AV | replaced → correct | 655 | 0.3361 | 10.000 | 9.932 | 143 | 80 |
| TV | empty → correct | 655 | 0.1156 | 1.587 | 1.505 | 20 | 10 |
| TV | replaced → correct | 655 | 0.0357 | -0.317 | -0.302 | 7 | 9 |
| ATV | empty → correct | 655 | 0.0328 | -0.159 | -0.256 | 10 | 11 |
| ATV | replaced → correct | 655 | 0.0296 | -0.159 | -0.155 | 7 | 8 |

Correct means original history, not a guarantee of a correct prediction. Positive ΔMSE means real history lowers MSE. Corrections/harms use the same paired nonneutral samples, not independently evaluated supports.
Empty/replaced states can be out of distribution. This test does not identify pure historical semantics or establish a natural-language causal effect. No new method, gain, or deployment result is established by running the audit.
