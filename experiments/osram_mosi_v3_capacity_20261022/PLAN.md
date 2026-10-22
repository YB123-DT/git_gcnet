# MOSI smaller Flat / larger V3 finite implementation plan

Goal: execute six seed66 MOSI150 capacity configurations, preserving CHM and
new GCNet missing. Reuse existing switches; do not modify any model formula.

## Registered design

| Mode | Flat hidden | V3 width | PairMLP hidden |
|---|---:|---:|---:|
| f512_plain | 512 | off | off |
| f512_v128 | 512 | 128 | 256 |
| f512_v256 | 512 | 256 | 512 |
| f1024_plain | 1024 | off | off |
| f1024_v128 | 1024 | 128 | 256 |
| f1024_v256 | 1024 | 256 | 512 |

All outputs1600; original LocalSkip,8x64 CHM,causal read-before-write,joint write
and step0.6 stay unchanged. Flat means its intermediate width, not output width.
From scratch joint150epochs,LR3e-4(1..100)/3e-5(101..150),batch32,Adam,WD1e-5,
MSE,original dropout,cyclic0..0.7,gcnet-random-mask-v1,official MOSI split,s0.
Per-rate TEST-best W-F1,matching ACC/MSE/predictions/masks,internal diagnosis only.
Reference original Flat66 and defaultV366 are reused, not retrained.

The user authorized running several combinations without another confirmation.
No seeds67/68,HPO,new losses,modules or checkpoint deletion are included.

## Files and steps

- [x] Worker and configuration tests: first fail on missing six-arm config,
  then implement strict configuration whitelist, same-width Flat equality,
  reference mask/init/data/LR parity, two-epoch real smoke and full persistence.
- [x] Queue and report tests: first fail on missing worker commands/reservation
  admission; implement finite persistent queue and completed-only report.
- [x] Run focused tests with existing V3 capacity regression tests; py_compile
  and diff check. Inspect effective parameter counts and initialization.
- [ ] Commit source, deploy a git archive to an immutable remote directory.
- [ ] On two healthy idle biggpu GPUs, one smoke per GPU at a time. Each smoke
  performs four actual MOSI optimizer steps because both old adapter and V3
  decoder are initially zero. Test full optimizer/RNG/prediction restoration.
- [ ] Admit at most three formal jobs per GPU from measured peaks with headroom;
  queue otherwise. Never reduce batch/width or signal another task.
- [ ] Confirm formal processes, GPU mapping and first epochs; commit/push small
  launch/validation evidence. Leave the finite queue to complete six jobs and
  automatically summarize all results, including failures.

## Scientific and operational limits

This is capacity allocation screening, not a parameter-matched proof of V3.
Compare each V3 arm with its own same-width Flat control. Results are seed66
Test-oracle only. Preserve all checkpoints,exact masks,predictions,last states.
Existing MOSEI and classification tasks use their immutable sources untouched.
Only new-worker first-two-epoch V3 hook instrumentation may be limited after
gradient connectivity is proved; do not alter training math or RNG.
