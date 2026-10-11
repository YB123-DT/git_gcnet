# Counterfactual Evidence Readouts

INTERNAL DIAGNOSTIC ONLY — per-rate BEST Test-oracle.

Approved scope: MOSI seed66, 150 epochs, two methods only.

- `marginal`: original Flat plus CEMR leave-one-evidence-out effect encoder.
- `coalition`: original Flat plus CIR singleton/pair Möbius effect encoder.

Both run one OSRAM trajectory, preserve the trainable original Flat anchor, use
only the original emotion MSE, and add a zero-initialized readout residual. All
Flat subset views share one training-time Dropout mask per utterance.

Protocol: cyclic random missing, rates0.0--0.7, first100 epochs at LR.001 then
full-state continuation to150 at LR.0001, per-rate Test-oracle selection. No
JEPA, completion, Gate, Nested, paired views, persistent mix, auxiliary loss or
Memory modification. Results remain pending until both verified runs complete.

## Engineering verification

Remote targeted suite: 231 passed. One unrelated legacy positional-configuration
test fails on the pre-existing optimizer/lr-schedule field ordering; the new
readout tests, runner tests, Base-first tests (apart from its archive-only Git
fixture), coalition tests, and all other selected missing-M3 regressions pass.

Real MOSI batch, three Adam updates per mode:

| Mode | Parameters | Peak CUDA | Three losses | Branch gradient L1 nonzero |
|---|---:|---:|---|---|
| marginal |14,187,105|2.003 GiB|3.043 / 240.198 / 17.953|yes|
| coalition|14,187,873|3.228 GiB|3.043 / 230.984 / 21.732|yes|

Losses are initial engineering smoke values, not performance evidence. Both
jobs together have a measured standalone peak sum of5.231GiB, leaving more than
26GiB on one32GiB V100 before launch. This supports single-card concurrency
without changing batch size or protocol.
