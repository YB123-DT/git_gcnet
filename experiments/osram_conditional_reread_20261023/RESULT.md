# Conditional second Memory read: startup record

INTERNAL TEST-ORACLE SCREEN

## Implemented and verified

The free/span pair reads the same actual alpha-decayed pre-write CHM after
original Base/Gap reads. It reuses one computed Local and performs no second
scan or changed write. Only forward Base receives the zero-initialized output
projection; original Gap, Local Skip, Flat1600, normalization/head and MSE remain.

The span control uses only original Base and currently missing targets' actual
effective Gap queries, with a differentiable pseudoinverse. Both variants have
identical initialization and parameter counts. New parameters: **429,062**;
total: **13,938,855**, unchanged Flat Adapter: **9,535,104**.

CPU: **67 focused tests passed** (including old V3 regressions). All four
real-data GPU smokes passed four actual updates, finite/nonzero new-parameter
gradients, exact initial Flat output and complete model/optimizer/RNG/output
reload. See `VALIDATION.md` and `SMOKE.md`; these checks are not task gains.

## Actual finite jobs

All four formal worker processes were launched on `ssh biggpu`, physical GPU1,
UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`. At the timestamp in
`LAUNCH.json`, all are alive in preflight/initialization; no formal scores are
available yet. Physical GPU4 is excluded. Existing GPU2/3 tasks were not touched.

| Dataset | Variant | Seed | Budget | PID |
|---|---|---:|---|---:|
| MOSI | free | 66 | 150 epochs: 3e-4 through 100, then 3e-5 | 86047 |
| MOSI | span | 66 | same | 88522 |
| MOSEI | free | 66 | 60 epochs: constant 1e-3 | 87276 |
| MOSEI | span | 66 | same | 89731 |

All use original GCNet random mask v1, cyclic0–0.7, batch32, Adam, WD1e-5,
original data/splits/s0 environment and original Flat66 exact test masks.
Per-rate TEST-best W-F1 is internal exploration; different rates may use
different selected epochs, not a single deployable checkpoint.

Source commit: `8f15f5315e1bd23fa89765d4a8344128719858e7`.
Runtime snapshot:
`/data2/yb/remote_experiments/osram_conditional_reread_20261023/source_8f15f53`.
Outputs:
`/data2/yb/remote_experiments/osram_conditional_reread_20261023/execution_v1`.
Persistent controller: tmux socket `osram_reread`, session `seed66`, PID40242.

## Interpretation reserved until completion

The controller automatically trains the four authorized jobs and summarizes
completed-only eight-rate/high-missing and per-rate ACC/W-F1/MSE. No extra
seeds, hyperparameter search, query rounds or losses are scheduled. All BEST,
matched predictions/masks and full last states are retained in independent
outputs. No baseline/checkpoint was deleted or overwritten.

The primary architectural comparison is **free minus span**, not just versus
Flat. If free does not outperform span, this implementation does not support
the new-address-access explanation. Equal gains cannot establish that new
directions caused an improvement. Joint training changes upstream states;
this is not a fixed-Memory information-theoretic causal experiment. Numerical
rank cutoffs can be nonsmooth. No performance conclusion is available yet.
