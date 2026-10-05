# Nested GNN — root-aware pooling confirmation

INTERNAL DIAGNOSTIC ONLY; per-rate BEST Test-oracle, not formal validation-selected paper results.

## Locked change

User approved the three-seed variant after reviewing old/new pooling difference.
Original model: nested_gnn_rooted_evidence, three-seed mean80.488818%,
high75.251816%, historical model commit ad211c0. Flat reference80.559048%,
high75.593862%. Implementation parent d7d7723.

New ID: nested_gnn_rootaware_evidence. Each GIN-style layer's subgraph pooling:

```
old: mean(all subgraph nodes)
new: shared Linear(128,64)(concat(root hidden, mean(non-root hidden)))
```

Neighbor vector is exactly zero for a singleton. One projection reused for
all roots/all three layers; no additional activation, attention, gate or loss.
All original graph edges, head tokens, root/distance embeddings, message layers,
three-layer concatenation pool and outer readout retained. New initialization
forks RNG after common parameters; original method remains unchanged.

Actual placement is Flat ADAPTER INPUT ONLY: residual changes its Local and
forward Base/active Gap inputs. Original Local Skip is NOT modified. Backward
constant-zero half, OSRAM Memory read/write/query, classifier and task MSE stay
unchanged. Zero-init decoder bridges preserve original Flat at initialization.

Measured counts: Flat13,509,793; old Nested13,669,028; new13,677,284.
Added over Flat167,491; added over old8,256. No parameter-count claim of lower
runtime: every root still runs a three-layer subgraph network.

## Verification

13 CPU tests passed on biggpu s0 (11.86s), one existing torch-geometric warning.
Tests-first red verified unsupported runner variant and absent root-aware API.
Focused tests cover exact root/neighbor concatenation, singleton zeros, shared
projector/count, old output/parameters/RNG, inactive/padding NaN safety,
initial identity, upper-half preservation, first utterance, finite gradients,
actual parameter updates and real full-model task training3 steps. Old40catalog
remains40; variant registered separately. Missing-comparator reporting also
tested; means are not computed from incomplete seed sets. Specification and
quality reviews passed. No extra GPU smoke/dependency.

## Authorized runs

MOSI seeds66/67/68 each from scratch100epochs, original Adam/lr/batch32/MSE,
no-JEPA cyclic random missing .0-.7. Same-seed Flat references and original
evaluation masks, per-rate BEST Test-oracle. No loss/width sweep, no persistent
mix, completion, paired views or frozen modules. Old Nested/Flat results reused.

Server biggpu GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153 only;
GPU4 forbidden. Root /data2/yb/remote_experiments/osram_nested_rootaware_20261005.
`dispatch.py` admits three jobs with live memory/disk checks and independent
logs/artifacts; no unrelated process termination or batch modification.

Run command, from sealed source with existing s0 Python:

```
python -m experiments.osram_nested_rootaware_20261005.dispatch \
  --root /data2/yb/remote_experiments/osram_nested_rootaware_20261005 \
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
```

Each run stores source/data/config provenance, full last_training.pt, history,
8 BEST checkpoints and 8 prediction arrays. Final runner checks100epochs,
artifact presence, snapshot hashes and exact canonical evaluation mask hashes.
After all three finish, SUMMARY.json includes per-rate/per-seed scores and
matched old Nested/Flat, sample SD, mean8/high. Failed/missing runs are retained
without blind retry or silently averaging only successful seeds.

Status: all three completed100epochs, exit0/outputs_verified=true, from sealed
code227dfe2 on biggpu GPU6. Each retained8 BEST files,8 prediction arrays and
full last_training.pt. Ordered evaluation mask verification passed in runner.

## Completed three-seed results

W-F1 (%), equal seed/rate weights, original per-rate BEST Test-oracle.
SUMMARY.json is the archived raw per-seed/per-rate comparison.

| Seed | Flat mean8 | Old Nested mean8 | Root-aware mean8 | Delta vs old |
|---|---:|---:|---:|---:|
|66|81.068|80.992|80.177|-0.816|
|67|80.556|80.851|80.082|-0.768|
|68|80.053|79.624|79.520|-0.103|
|Mean|80.559|80.489|79.926|-0.562|

| Metric | Flat | Old Nested | Root-aware | Delta vs old | Delta vs Flat |
|---|---:|---:|---:|---:|---:|
|Mean8|80.559|80.489|79.926|-0.562|-0.633|
|High .5/.6/.7|75.594|75.252|75.016|-0.236|-0.578|
|Mean8 sample SD|0.507|0.752|0.355|—|—|

| Rate | Flat | Old Nested | Root-aware | Delta vs old |
|---|---:|---:|---:|---:|
|0.0|88.419|88.114|87.563|-0.551|
|0.1|85.845|85.925|85.596|-0.329|
|0.2|83.431|84.456|82.631|-1.824|
|0.3|80.999|80.760|80.617|-0.144|
|0.4|78.996|78.900|77.958|-0.943|
|0.5|77.327|76.843|75.817|-1.027|
|0.6|75.848|75.780|75.720|-0.060|
|0.7|73.607|73.132|73.511|+0.379|

All three seeds have lower overall mean8 than original Nested. This shared
root/non-root pooling variant fails to improve the chosen aggregate. The sole
rate.7 mean increase does not offset declines elsewhere. Do not infer a failure
mechanism from these aggregates. Retain old implementation/results; no further
training/configuration search is automatically launched. These remain internal
adaptive Test-oracle diagnostics, not formal paper evidence of generalization.

## Historical launch

All three started on biggpu GPU6 from sealed code227dfe2;
dispatcher PID2658634, tmux osram_nested_rootaware_20261005.
seed66 PID2663369 completed2 epochs; seed67 PID2684453 completed1 epoch;
seed68 PID2747902 loaded data at launch inspection. No traceback found.
LAUNCH.json records actual commands, source, GPU UUID and PIDs at launch.
