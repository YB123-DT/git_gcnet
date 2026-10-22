# KV association fit and future sentiment value

INTERNAL DIAGNOSTIC ONLY

## Completed scope and provenance

MOSI seed66, the protected original cyclic no-JEPA LargeFlat miss0.3 BEST
(epoch50), frozen across all eight GCNet random missing rates0.0–0.7.
This is one checkpoint, not eight separately selected/independently trained
models. Checkpoint SHA256:
`fd42eeaba7a7f24a748c6e3cf514f8b2cc8afe58cfc5b90968e0514fcb8aadc3`.
Exact original batch test-mask bank SHA256:
`4cae181f46efbb13a4c7200b0d295dea8052a08871b25a50399f9a0b351ffdbb`.

Numerical source: `d453f75fc9f506add7a0bdac4e14065b3c0ef6c2`.
biggpu physical GPU2 / `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`;
unchanged MOSI s0 environment, Torch2.2.2+cu121, NumPy1.24.3, sklearn1.3.2.
No optimizer or new training steps. No production source changes. No running
jobs stopped, moved, or reconfigured.

## Main result

31 test conversations, 686 valid utterances per mask rate, 5240 distinct
conversation/rate/source-write events with an eligible future target, and
18232 source-write/target pair slots. The same utterance/write can appear in
multiple intervention records; these are not independent test examples.

| Measurement | Result |
|---|---:|
| Mean normalized joint KV error before write | 0.441689 |
| Mean normalized joint KV error after write | 0.071120 |
| Mean Q = before − after | 0.370570 |
| Pair slots with Q > 0 | 18232 / 18232 |
| Spearman(Q, future task utility) | −0.016252 |
| Conversation-cluster bootstrap95% interval for Spearman | [−0.042046, +0.010223] |
| Mean U = MSE(omit) − MSE(keep) | +0.000857 |
| Conversation-cluster bootstrap95% interval for mean U | [−0.005291, +0.009108] |
| U > 1e−7: keeping write helps regression | 9007 |
| U < −1e−7: keeping write harms regression | 9225 |
| Polarity corrections from keeping write | 184 |
| Polarity harms from keeping write | 183 |

Bootstrap uses200 resamples of entire conversations, seed66; this describes
the frozen audit, not checkpoint/seed or independent-generalization uncertainty.
Q pools the actual sqrt(beta)-weighted active joint K/V columns and heads;
both errors use the same value-energy denominator. It is not an emotion metric.

## Future distance

| Future offset | Pair slots | Mean U | Spearman(Q,U) | Polarity corrections | Polarity harms |
|---|---:|---:|---:|---:|---:|
| 1 | 5240 | +0.005263 | −0.014028 | 112 | 100 |
| 2 | 4992 | −0.004090 | −0.039662 | 48 | 57 |
| 4 | 4496 | +0.002657 | −0.000835 | 21 | 23 |
| 8 | 3504 | −0.000996 | −0.005407 | 3 | 3 |

All four offsets have weak unadjusted rank association; none establishes a
useful monotonic proxy. Counts are paired intervention slots, not an aggregate
deployment ACC/W-F1 difference. Neutral labels enter MSE, not polarity counts.

The descriptive Q-quartile mean utilities are −0.000457, +0.004190,
+0.005577, −0.005886. Thus larger immediate KV improvement did not monotonically
coincide with larger future task benefit in this audit. Quartiles are not
learned thresholds or causal mediation evidence; group-specific sampling and
uncertainty remain relevant. Full rate, availability, distance×availability,
quartile and median-split tables are in `completed/full/`.

## Replay checks and cost

- 26 local core/report/legacy-retention tests passed, including distances1/2/4/8,
  padding, zero-write-step control, and a deliberately changed mask bit.
- Real MOSI GPU smoke: one batch,686 utterances,2279 pairs,17.91seconds,
  peak allocated CUDA676.83MiB. This is correctness evidence, not performance gain.
- Full frozen eight-rate audit:69.25seconds, peak allocated CUDA678.97MiB.
- Normal replay pre-read Memory maximum absolute error: **0**, all eight rates.
- Normal prediction and direct full-model omit replay maximum absolute error:
  **7.152557373046875e−7**. No factual polarity changes from replay.
- Source current prediction unchanged; full saved miss0.3 prediction/label/
  availability parity; original bank hashes checked at all eight rates.
- Model parameter digest unchanged, no accumulated gradients; checkpoint file
  hash unchanged after execution. No writes with near-zero value energy.
- Independently rechecked all18232 row keys, distances, Q arithmetic, scalar
  MSE, utility signs, polarity corrections/harms, total coverage and row SHA256.

Omit skips exactly one source joint write, keeping its alpha decay and current
read. All subsequent writes remain normal joint A/T/V writes; original captured
Local/queries and classifier are used. This includes the full coupled write's
effect and future state evolution, not an isolated modality/Value contribution.

## Conclusion and limits

The original write substantially fits its current internal K/V associations,
but the size of that immediate improvement does **not show a clear positive
association** with future sentiment utility in this fixed-checkpoint audit.
KV fitting quality therefore is not supported here as a sufficient proxy for
task-relevant write value.

This does not prove CHM is useless or the write equation is defective. Individual
write removals can be compensated by existing/subsequent history; removing one
write is different from removing all Memory. Value norm, write-update magnitude,
source/target conditions and history position may confound associations. Current
grouping is descriptive, not causal attribution or proof of an improved writer.
No new module/training or automatic sweep is authorized by these results.

## Saved evidence

Remote output:
`/data2/yb/remote_experiments/osram_kv_task_utility_20261022/execution_v1/`.
Raw scalar rows remain there, SHA256:
`d9746d9762ba37760c75e9ca2bd80aad731416244e63f9365cc986d4065981db`.
No checkpoint, feature cache or large per-sample archive is added to Git.
Tracked `completed/full/STATUS.json`, `REPORT.json`, `AGGREGATE.csv`, `RESULT.md`
and smoke status preserve the measured metadata and all summary groups.
