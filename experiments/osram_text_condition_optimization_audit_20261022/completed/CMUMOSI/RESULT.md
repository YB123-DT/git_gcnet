# CMUMOSI Text-condition optimization audit

INTERNAL DIAGNOSTIC ONLY

One fixed original Flat66 miss0.3 checkpoint. No new training. Temporary SGD is fully restored.
Test stratification uses exact eight-rate GCNet masks; gradients/updates use TRAIN labels only.

| Test group (pooled slots) | N | Local WF1 | Full WF1 | Corrections | Harms | LocalMSE-FullMSE |
|---|---:|---:|---:|---:|---:|---:|
| overall | 5488 | 75.434780 | 77.870007 | 392 | 249 | 0.175062 |
| Textvisible | 3551 | 86.616754 | 86.539329 | 52 | 52 | 0.058290 |
| Textmissing_seen | 1746 | 53.822764 | 62.344486 | 329 | 186 | 0.426948 |
| Textmissing_never | 191 | 57.074649 | 58.311930 | 11 | 11 | 0.043468 |

| Gradient group | Nonzero pairs | Mean cosine | Negative fraction |
|---|---:|---:|---:|
| overall | 64 | 0.426214 | 0.250000 |
| ObservedEncoder | 64 | 0.232661 | 0.250000 |
| WriteKey | 64 | 0.303996 | 0.250000 |
| WriteValue | 64 | 0.358531 | 0.250000 |
| Query | 64 | 0.280318 | 0.250000 |
| Memory | 64 | 0.281274 | 0.312500 |
| LocalFlatHead | 64 | 0.466998 | 0.250000 |
| Other | 64 | 0.361497 | 0.250000 |

| Temporary update | Mean cross-condition deltaMSE | Harms | Improvements | Neutral |
|---|---:|---:|---:|---:|
| visible_to_missing | -0.02492435 | 17 | 47 | 0 |
| missing_to_visible | -0.02419181 | 16 | 48 | 0 |

## Limits

- One seed66 fixed miss0.3 Test-oracle checkpoint,not per-rate deployed performance.
- Eval-mode gradients remove dropout noise; they are not stochastic training trajectories.
- Paired prefixes encode past/current separately with the original utterance-local encoder to fix historical GEMM shapes; test replay remains unmodified.
- Temporary SGD is not Adam or persistent optimization; same-pair finite effects are local sensitivity.
- MSE residual signs can reverse loss gradients even with aligned prediction Jacobians; inspect both.
- Test readout zeroing may be out of distribution; history benefit is model response,not information quality.
- Pairs are label-free current-Text interventions on legal GCNet background masks,not fresh performance samples.
- Repeated conversations across rates are not independent samples; no significance/long-term transfer claim.

Opposite MSE residual signs: 0.250000; prediction-Jacobian mean cosine: 0.8609627666657141.
