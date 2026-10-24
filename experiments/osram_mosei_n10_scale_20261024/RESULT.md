# Frozen MOSEI N10 correction strength — completed

INTERNAL DIAGNOSTIC ONLY

Conclusion: halving correction barely changes eight-rate W-F1 (+0.001793pp),
with a small high-missing change (+0.056794pp). It does not exceed the independently
trained Flat128 seed66 W-F1. This does not support a substantial strength-only fix.

## Protocol

Completed N10 seed66:Flat128, Nested dim64/depth3/groups2/MLP128, cmerc-repro,
new GCNet missing. Hold each rate’s existing TEST-best checkpoint and exact mask
fixed across gamma0/.5/1. Scale all five task-input correction slots together:
`X_raw + gamma*(X_nested-X_raw)`. Original LocalSkip, CHM reads/writes/queries,
head, inactive gaps and zero backward halves remain unchanged. No training or
coefficient-dependent checkpoint selection. Gamma0 is the N10-trained checkpoint
with correction bypassed, NOT the separately-trained Flat128.

## Eight-rate and high-missing metrics

| Gamma | ACC | W-F1 | High W-F1 | ΔW-F1 vs gamma1 (pp) |
|---|---:|---:|---:|---:|
| 0.0 | 83.623418 | 83.366358 | 80.016609 | -0.011104 |
| 0.5 | 83.616538 | 83.379255 | 80.013354 | 0.001793 |
| 1.0 | 83.592460 | 83.377462 | 79.956560 | 0.000000 |

Separate Flat128 seed66 reference:ACC 83.606219,
W-F1 83.387044, high W-F1 80.011427.
Halved correction remains0.007789pp below that Flat W-F1.

## Per-rate W-F1

| Missing rate | gamma0 | gamma.5 | gamma1 |
|---|---:|---:|---:|
| 0.0 | 87.273697 | 87.335132 | 87.369885 |
| 0.1 | 86.652528 | 86.667022 | 86.699260 |
| 0.2 | 85.254951 | 85.341783 | 85.402157 |
| 0.3 | 84.769987 | 84.626282 | 84.561335 |
| 0.4 | 82.929873 | 83.023757 | 83.117375 |
| 0.5 | 81.879116 | 81.838195 | 81.778345 |
| 0.6 | 79.716257 | 79.700466 | 79.739347 |
| 0.7 | 78.454455 | 78.501402 | 78.351989 |

No coefficient is uniformly superior. Gamma1 is better at rates0/.1/.2/.4/.6;
gamma0 is better at.3/.5; gamma.5 is better at.7. These are frozen sensitivity
results on exploratory test-selected weights, not a deployable per-rate oracle.

## Paired classification counts

All counts below pool eight rates, repeating utterances. They are not unique
utterance counts or W-F1 differences. Gold0 is excluded only for polarity counts.

Against gamma1:
- gamma0:91 corrections/82harms overall; A-only positives25/0, A-only negatives0/11.
- gamma.5:49corrections/42harms overall; A-only positives16/0, A-only negatives0/7.

Against independently trained Flat128 seed66:
| Gamma | A-only positive corrections | A-only positive harms |
|---|---:|---:|
| 0 | 46 | 99 |
| .5 | 45 | 107 |
| 1 | 43 | 121 |

Reducing correction restores some A-only positives but loses negative correctness.
Most positive harms remain even with correction off. Cross-model losses therefore
cannot be explained solely as excessive direct correction; remaining differences
include jointly trained main-path parameters and different selected epochs.
This does not separately identify Memory writing, reading or classifier drift.

## Verification, cost and provenance

- Six CPU unit tests passed locally and in cmerc-repro; initial red tests observed.
-131 numerical model/sampler files unchanged from the N10 source snapshot.
-176 Memory forwards:22batches per rate, one scan reused by three readouts.
- Gamma1 reproduces original saved predictions within1e-6 with exact polarity;
  all24 output prediction hashes and all eight original metrics checked independently.
- Model parameters and all original checkpoint hashes unchanged; readout replay
  does not advance CUDA RNG. No optimizer/update is created.
- Replay wall time:76.748seconds including setup; physicalGPU1.
- Implementation source:`a92dfacdad22345f5235765a09095da4369c0f7e`.
- Remote output:`/data2/yb/remote_experiments/osram_mosei_n10_scale_20261024/replay_v2`.
- Original remote SUMMARY SHA256:`ce2a5a1a810019b9fdf0e6910b6951223c49ccd31a5fe2990bfa9f2f025787ec`.
- Detailed config, per-rate checkpoints/predictions, all cohort counts, environment
  and source manifest are in `results/SUMMARY.json`; score table in `results/PER_RATE.csv`.
- Failed replay_v1 is preserved:pre-inference tuple/list configuration comparison
  mismatch. Regression fix canonicalizes JSON representation without relaxing
  epoch/actual-value checks. source_v2/replay_v2 are independent.
- All protected checkpoints, prior experiments and the active training queue were
  untouched. No further scales, methods, training or seeds were added.
