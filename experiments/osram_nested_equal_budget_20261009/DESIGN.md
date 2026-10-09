# Near-equal parameter budget: small Flat and wider-state Nested

Status: DESIGN ONLY. No model implementation changes, registration or training launched.
INTERNAL DIAGNOSTIC ONLY for historical source scores. Counts are not performance results.

User approved exploring Flat256 while moving the original large Flat parameter budget into Nested. Keep the original mechanism, not just a very wide MLP returning64-dimensional states. The latest scope is to determine a concrete configuration before launch.

## Verified budget

Source baseline metrics: biggpu `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/metrics.json`, parameter_count13,509,793.
Small Flat256 without Nested:5,508,961. Therefore exact module budget8,000,832.
Instantiate existing `TokenAdapter(NestedSweepGNN(dim=d, mlp_hidden=h),256,8,64,d)` on CPU; sum all parameters, no random data inference or training.

| Graph state d | Internal MLP hidden h | Nested parameters | Whole model with Flat256 |
|---|---:|---:|---:|
|64|64|159235|5668196|
|256|256|1322755|6831716|
|512|512|4479747|9988708|
|704|704|8051715|13560676|
|512|1000|7980171|13489132|

Whole model is derived from measured module count plus the existing5,508,961 non-Nested small Flat model, not a newly trained model measurement. Before implementation/run, verify the instantiated full model count.

## Recommended locked configuration

- Flat remains4352->256->1600; Local Skip and task head unchanged.
- OSRAM remains8heads,64-dimensional key/value; no write/query/history/mask changes.
- Nested graph state64->704: each real64-dimensional memory-head read projects to704; Local256 projects to704.
- Same node count, evidence roles/head identities, root-induced topology,3GIN layers and Tanh. Do not add independently parameterized root networks.
- GIN MLP704->704->704, followed by existing LayerNorm704.
- Root summary concatenates3layers:2112->704->704.
- Graph readout concatenates original node/root summary/global mean:2112->704->704.
- Local decoder704->256, eight shared-across-evidence head decoders704->64. Final decoders remain zero initialized; original input residual and empty-history/padding masks unchanged.
- No auxiliary loss, extra attention, completion, JEPA or freezing.

Nested704 partition: tokenizer557,568; core6,952,707; Local decoder180,480; memory decoders360,960. Most parameters are in the original graph core, not a newly appended dense head.
Whole-model delta versus original large Flat:+50,883 (+approximately0.377%). Call this NEAR-equal, not exactly equal. The rounding choice704 is11x64; it is only an implementation-friendly width, not a meaningful new head count or semantic choice.

## Alternatives and rationale

1. Recommended: state704 and hidden704, one coordinated width change preserving the original h=d relationship.
2. State512 with MLP hidden1000: closer count (−20,661, approximately−0.153%), but changes two capacities and uses an arbitrary hidden width to match the budget. Valid alternative, not another authorized run.
3. Keep state64 and expand only its MLP to reach8M: not recommended; preserves64-dimensional communication bottleneck and principally enlarges update networks. Prior256/512 MLP-only expansions failed to improve the small-Flat combination; they do not prove wider states must fail.

## Future verification and interpretation (not launch authorization)

Check full-model parameter count, exact unchanged outer initialization/RNG, zero-init prediction parity with Flat256, inactive evidence/padding/empty-history behavior, finite gradients and genuine updated graph weights. Reuse original tests, not a new broad test campaign.
Profile one representative actual batch on the assigned healthy biggpu device before choosing concurrency: equal parameter count is not equal compute/activation memory, because shared GINs execute in multiple rooted subgraphs. Never use GPU4.
Compare against original large Flat and existing small Flat256+original Nested, preserving data/seed/training budget/selection protocol when a run is requested. Do not interpret a single architecture outcome as proving where all parameter budgets should be spent.
