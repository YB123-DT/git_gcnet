# Stage-aware paired mask protocol design

## Problem

The current fixed bank assigns one missing-pattern matrix to every utterance and
reuses it for all 100 training epochs, validation, and test. This departs from
the reproduced GCNet path, where training resamples patterns at the same missing
rate. The change reduced Original IEMOCAP-6 weighted F1 from 0.6089 to 0.5782 at
missing 0.5 and from 0.6112 to 0.5512 at missing 0.7 for seeds 66--70.

## Selected design

Keep the requested missing rate fixed. Pre-generate a deterministic bank for
each training epoch and two independent banks for validation and test:

```text
train/epoch_001 ... train/epoch_100
validation
test
```

All arms with the same rate and seed consume identical stage/epoch banks. A
training utterance therefore sees changing missing patterns across epochs while
validation and test remain fixed and reproducible. The 0.7 generator retains
the legacy one-observed-modality behavior, whose realized missing rate is 2/3.

## Interfaces and invariants

- `mask_bank.py` creates and hashes a stage-aware bundle from a master seed.
- Train masks are selected by one-based epoch; validation and test masks are
  selected by stage and never by call order.
- Every constituent bank uses the existing legacy mask generator.
- Original, Full, and CP-LECC receive exactly the same bank for a paired job.
- Saved provenance records the bundle hash and constituent hashes.
- Model, graph topology, optimizer, losses, fold 5, epochs, and features remain
  unchanged.

## Verification and experiment gate

Unit tests lock deterministic regeneration, epoch diversity, cross-arm reuse,
stage isolation, and 0.7 realization. First run only Original for rates 0.5 and
0.7 with seeds 66--70 on biggpu's `gcnet-official` environment. Continue to
Full and CP-LECC only if the recovery run is materially consistent with the
August 19 reproduction: no more than 1.5 weighted-F1 points below its same-seed
mean at either rate. The old single-fixed-bank grid remains diagnostic evidence
and is not overwritten.
