# Fixed-rate validation

INTERNAL TEST-ORACLE SCREEN. No model/mask/core-trainer changes.

- Worker tests were written before implementation and failed because the
  new worker did not exist; subsequent23 worker/queue tests passed(0.41s).
- Fixed config changes only3 fields relative to the accepted cyclic config;
  unsupported rates/seeds/budgets, added blocks and unexpected changes reject.
- Same-rate selection uses strictly higher W-F1, preserves earliest ties,
  and does not substitute ACC/MSE for selection.
- Eight unique jobs; resource reservations; incomplete results excluded.

Real-data GPU smoke: biggpu physicalGPU2, s0, fixed0.7, one full MOSI epoch,
2 real optimizer updates. Completed in13.891seconds; peak allocated1091.943MiB.
Actual masks were intercepted WITHOUT altering the original preparation path:
rate counts `{"0.7":2}`,1284 valid utterances,414 A-only/420 T-only/450 V-only,
zero empty/pair/full patterns. Actual missing fraction2/3 is the required
original GCNet high-missing one-hot branch, not a changed0.7 sampler.

Original Flat13,509,793 parameters and fresh initialization digest exactly
matched cyclic Flat66. Actual nonfinite loss/gradient/mask checks passed;
parameters updated. Full last checkpoint restored model/Adam/RNG and exact
test predictions, with unchanged accepted corresponding test mask.

Smoke output:
`/data2/yb/remote_experiments/mosi_fixed66_smoke_dTL0p3/rate_0p7`.
The smoke used a working-tree copy based on7edbd72, not a claimed committed
formal training snapshot. Its numerical-source hashes are recorded in
`SMOKE.json` and checked against local source before formal snapshot creation.
It proves implementation functionality, not sentiment performance.
