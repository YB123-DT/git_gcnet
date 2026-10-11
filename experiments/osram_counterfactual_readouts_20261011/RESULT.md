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
Memory modification.

## Final seed-66 result (150 epochs)

| Method | miss=.0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 | 8-rate mean | High (.5/.6/.7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cfg84 Flat reference | 88.205 | 86.507 | 83.187 | 80.763 | 80.827 | 77.494 | 75.790 | 75.773 | 81.068 | 76.352 |
| Marginal | 87.237 | 86.138 | 82.396 | 81.942 | 79.224 | 76.854 | 76.112 | 76.241 | 80.768 | 76.402 |
| Coalition | 86.706 | 84.973 | 81.931 | 81.267 | 79.730 | 75.986 | 76.073 | 75.232 | 80.237 | 75.764 |

Relative to Flat, Marginal changes the 8-rate mean by `-0.300` percentage
points and the high-missing mean by `+0.050` points. Coalition changes them by
`-0.831` and `-0.589` points respectively. Neither method improves the primary
8-rate aggregate in this single-seed internal screen.

Marginal selected epochs by rate were `89/66/65/93/75/70/64/70`. Coalition
selected `62/131/84/71/67/62/67/79`; only Coalition miss=.1 selected an epoch
after 100. Evaluation-mask hashes match the Flat reference at all eight rates.

## Marginal seed67 follow-up

Status: running. The follow-up uses the same 150-epoch schedule and changes only
the seed-specific cfg84 reference from seed66 to seed67. It runs from immutable
source commit `7d5c345` on biggpu GPU0 as PID `4026922`; results are not yet
available.

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
