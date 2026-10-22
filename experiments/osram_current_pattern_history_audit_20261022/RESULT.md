# Fixed-history current-pattern audit: completed

INTERNAL DIAGNOSTIC ONLY

MOSI seed66, one protected original CHM LargeFlat checkpoint: miss0.3 BEST,
epoch50, SHA256 `fd42eeaba7a7f24a748c6e3cf514f8b2cc8afe58cfc5b90968e0514fcb8aadc3`.
Source snapshot `83c0d0683abf0842193cd6bb10be413d4f2f3c34`. No new training,
optimizer, parameter update, teacher, or alteration to the original sampler.

## 1. Current-pattern historical Writer gradients

One deterministic label-independent target from each of 52 train conversations.
The seven current patterns share EXACT past features/masks/speakers, pre-write
Memory and past predictions (maximum error0). The current Key's direct
Gap-addressing derivative is excluded; only gradients through historical writes
reach the original K/V projector parameters. Target MSE remains the original task.

| Statistic | Writer | WriteKey | WriteValue |
|---|---:|---:|---:|
| Defined same-target pattern pairs | 1092 | 1092 | 1092 |
| Mean loss-gradient cosine | 0.3764 | 0.3364 | 0.3875 |
| Negative loss-gradient fraction | 22.253% | 22.802% | 22.161% |
| Mean prediction-Jacobian cosine | 0.6405 | 0.5785 | 0.6632 |
| Negative prediction-Jacobian fraction | 0.092% | 1.557% | 0% |

Writer loss gradients oppose for243/1092 pairs. In242 of those, the two scalar
predictions lie on opposite sides of the SAME sentiment label; prediction
Jacobians remain positively aligned in all242. Only1/1092 Writer prediction-
Jacobian pairs is negative. This does not erase genuine local task-gradient
competition, but does not support describing widespread incompatible Memory
representation directions either. Scalar MSE satisfies grad(loss)=2*(pred-y)*J.

Separately, average the gradients over these52 equal-weight targets FIRST, then
compare patterns (not the average of per-target cosines):4/21 pairs are negative:
A/AV=-0.0469, A/TV=-0.1011, T/AV=-0.0506, AT/AV=-0.0804. Remaining17 are positive.
These are gradients of this sampled target objective, not the actual entire
cyclic training distribution, Adam update directions, or long-term transfer.
All pair matrices and train MSE are in completed/RESULT.md and
completed/AGGREGATE_GRADIENTS.json.

## 2. Dependence on correct versus empty/replaced history

All686 original test utterances are replayed in seven current-only combinations.
The table below excludes31 first utterances with no history:655 valid utterances,
630 nonneutral polarity labels, SAME cohort in every row/condition.
W-F1 is percentage, threshold pred>0, gold0 excluded only from polarity metrics.

| Current | Correct history W-F1 | Empty W-F1 | Replaced W-F1 | Correct−Empty (pp) | Correct−Replaced (pp) |
|---|---:|---:|---:|---:|---:|
| A | 60.815 | 51.989 | 50.165 | +8.825 | +10.650 |
| T | 86.472 | 86.504 | 86.279 | -0.033 | +0.192 |
| V | 63.435 | 55.181 | 58.166 | +8.254 | +5.269 |
| AT | 86.535 | 86.626 | 85.701 | -0.092 | +0.833 |
| AV | 62.398 | 42.660 | 52.466 | +19.738 | +9.932 |
| TV | 86.659 | 85.155 | 86.961 | +1.505 | -0.302 |
| ATV | 86.844 | 87.100 | 86.998 | -0.256 | -0.155 |

| Current | Correct MSE | Empty MSE | Replaced MSE |
|---|---:|---:|---:|
| A | 2.31812 | 2.69192 | 3.20120 |
| T | 0.86473 | 0.96287 | 0.99669 |
| V | 2.34886 | 2.71567 | 2.53460 |
| AT | 0.84731 | 0.91938 | 0.96142 |
| AV | 2.33621 | 2.92514 | 2.67232 |
| TV | 0.93566 | 1.05128 | 0.97132 |
| ATV | 0.90818 | 0.94099 | 0.93781 |

MSE includes neutral labels. Correct history lowers MSE relative to both
controls in all seven conditions; polarity gains are predominantly no-Text.
MSE improvements must not be described as W-F1 improvements for T/AT/TV/ATV.
For AV, correct history versus empty produces142 corrections/57 harms; versus
replaced produces143 corrections/80 harms on630 identical polarity examples.

Replacement donors:31 deterministic other TRAIN conversations, sufficiently
long for every original strict prefix. Recipient availability, length and speaker
trajectory are unchanged; only history content is substituted. No donor selection
uses labels or performance. Original Local is EXACT across the three history
conditions; original Query/addressing/Flat/head are unchanged. No unavailable
donors. Correct means factual history, not a guarantee of correct prediction.

## Interpretation boundary

This checkpoint does use factual history substantially when current Text is
absent. The result does not support calling Memory globally useless or unused.
It also does not establish that a new readout/Writer will improve performance.
Loss-gradient competition is present but mostly accompanies residual-sign
differences, rather than broadly opposed prediction sensitivities.

These are independent current-only interventions on one original RANDOM history
trajectory(rate0.3), NOT Persistent Missing scores, eight-rate Test-best means,
separately trained mode-specific models, multi-seed evidence, or natural-language
causal effects. Empty/replaced states can be distribution-shifted. The fixed
checkpoint was originally Test-oracle selected. No downstream module follows
automatically from these observations.

## Verification, resources, delivery

- 23 targeted tests passed; compilation and diff checks passed.
- Real GPU smoke passed in17.70s, then full audit completed in148.85s.
- biggpu physicalGPU2, UUID`GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`, s0
  Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2; peak964.30MiB.
- Max original saved prediction error0; factual fixed-state replay error0;
  no-history boundary error0. Model digest unchanged; parameter.grad unmodified.
- 42 actual all/history-present metric rows independently recomputed with
  NumPy/sklearn agree (MSE atol1e-12; ACC/W-F1 atol1e-10).
- Initial GPU7 smoke failed after another PID transiently consumed31.16GiB.
  Failed evidence retained. Same source/protocol retried on live-checked GPU2;
  no other process signalled, no batch/architecture/mask changes.

Remote output:
`/data2/yb/remote_experiments/osram_current_pattern_history_audit_20261022/execution_v2/`.
Persistent runner: tmux socket`current_pattern_history`, session`audit_v2` (now
completed). Full STATUS/data provenance and raw prediction/gradient-pair streams
remain remote. No model checkpoint was created, changed or deleted.

Small deliverables: [full tables](completed/RESULT.md),
[machine-readable report](completed/REPORT.json),
[verification/source/config/mask hashes](completed/VERIFICATION.json),
[aggregate gradient geometry](completed/AGGREGATE_GRADIENTS.json),
[per-pattern metrics](completed/HISTORY_METRICS.csv).
