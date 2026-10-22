# CMUMOSEI Text-condition optimization audit

INTERNAL DIAGNOSTIC ONLY

One fixed original Flat66 miss0.3 checkpoint. No new training. Temporary SGD is fully restored.
Test stratification uses exact eight-rate GCNet masks; gradients/updates use TRAIN labels only.

| Test group (pooled slots) | N | Local WF1 | Full WF1 | Corrections | Harms | LocalMSE-FullMSE |
|---|---:|---:|---:|---:|---:|---:|
| overall | 37272 | 76.274792 | 82.893872 | 2642 | 1156 | 0.137017 |
| Textvisible | 24334 | 85.171030 | 86.922759 | 564 | 264 | 0.050458 |
| Textmissing_seen | 9511 | 48.967134 | 76.575160 | 1991 | 831 | 0.400914 |
| Textmissing_never | 3427 | 61.238870 | 66.490175 | 87 | 61 | 0.019248 |

| Gradient group | Nonzero pairs | Mean cosine | Negative fraction |
|---|---:|---:|---:|
| overall | 64 | 0.583852 | 0.171875 |
| ObservedEncoder | 64 | 0.275343 | 0.218750 |
| WriteKey | 64 | 0.521382 | 0.187500 |
| WriteValue | 64 | 0.535155 | 0.187500 |
| Query | 64 | 0.452000 | 0.171875 |
| Memory | 64 | 0.539604 | 0.187500 |
| LocalFlatHead | 64 | 0.600605 | 0.171875 |
| Other | 64 | 0.572110 | 0.203125 |

| Temporary update | Mean cross-condition deltaMSE | Harms | Improvements | Neutral |
|---|---:|---:|---:|---:|
| visible_to_missing | -0.00384305 | 11 | 53 | 0 |
| missing_to_visible | -0.00383290 | 11 | 53 | 0 |

## Limits

- One seed66 fixed miss0.3 Test-oracle checkpoint,not per-rate deployed performance.
- Eval-mode gradients remove dropout noise; they are not stochastic training trajectories.
- Paired prefixes encode past/current separately with the original utterance-local encoder to fix historical GEMM shapes; test replay remains unmodified.
- Temporary SGD is not Adam or persistent optimization; same-pair finite effects are local sensitivity.
- MSE residual signs can reverse loss gradients even with aligned prediction Jacobians; inspect both.
- Test readout zeroing may be out of distribution; history benefit is model response,not information quality.
- Pairs are label-free current-Text interventions on legal GCNet background masks,not fresh performance samples.
- Repeated conversations across rates are not independent samples; no significance/long-term transfer claim.

Opposite MSE residual signs: 0.171875; prediction-Jacobian mean cosine: 0.8683074789265469.
