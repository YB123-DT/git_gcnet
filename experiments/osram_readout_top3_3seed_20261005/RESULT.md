# Locked top3 one-stage readout confirmation

INTERNAL DIAGNOSTIC ONLY — per-rate BEST Test-oracle, not formal paper results.

User authorized three seeds for the top3 single-view one-stage modules.
Reuse completed seed66; six new seed67/68 runs, 100 epochs each. No new model
code, tuning, losses or baseline rerun. Original controllers, manifests, CPU
evidence/readiness and immutable model snapshots are reused with all hash
checks retained. Only seed, output/run identity and healthy device change.

| Method | Seed66 mean8 | Seed66 high | Original model commit |
|---|---:|---:|---|
| XCiT-XCA |81.041519|76.257088|9adf59f|
| Nested GNN rooted evidence |80.992147|76.077229|ad211c0|
| Neural Production |80.981287|76.475237|ad211c0|
| Flat reference |81.068095|76.352251|historical run commit unknown|

Server biggpu, GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
GPU4 forbidden. Staggered concurrent launch with live memory admission;
no unrelated process termination or batch change. No new GPU smoke.

Persistent command: existing s0 Python `dispatch.py --root
/data2/yb/remote_experiments/osram_readout_top3_3seed_20261005`.
DISPATCH.json records six PIDs/status/commands/output and original snapshots;
logs are separate. Original run verifies full100 history, eight BEST files,
predictions, masks, configuration, snapshot hashes and full last_training.pt.
SUMMARY.json is generated automatically once all six children finish, with
failed/missing runs explicitly retained and no substitution or automatic retry.

Status: all six runs completed100 epochs, exit0 and outputs_verified=true.
Original seed66 retained. Dispatcher PID
1984684, tmux `osram_readout_top3_3seed_20261005`. Launcher commit d14781b;
original model commits remain unchanged. Launch PID/status copy in LAUNCH.json.

| Method | Seed67 PID | Seed68 PID |
|---|---:|---:|
| XCiT-XCA |1985897|2043931|
| Nested GNN |2072790|2100965|
| Neural Production |2130521|2144888|

GPU6 had 17,765 MiB free after all six admissions. XCA67/68 completed8/6 epochs,
Nested67/68 completed3/2, Production67 completed1 at launch inspection;
Production68 was initializing at that earlier inspection. No traceback found.

## Completed three-seed confirmation

W-F1 (%), equal seed weights; per-rate BEST Test-oracle INTERNAL DIAGNOSTIC ONLY.
SUMMARY.json includes every seed/rate and all six process outcomes.

| Method | seed66 mean8 | seed67 mean8 | seed68 mean8 | Mean8 | High mean | Delta mean8 vs Flat |
|---|---:|---:|---:|---:|---:|---:|
| Original Flat |81.068|80.556|80.053|80.559|75.594|0|
| XCiT-XCA |81.042|80.037|79.561|80.213|75.365|-0.346|
| Nested GNN |80.992|80.851|79.624|80.489|75.252|-0.070|
| Neural Production |80.981|80.540|79.855|80.459|75.558|-0.100|

Nested high per-seed:76.077/75.497/74.181. Nested is closest on mean8 among
these three, but high mean remains0.342pp below Flat. Its mean8 gains at seed67
do not repeat at66/68. No method establishes an overall improvement. All eight
BEST checkpoints per new run are retained, with full training recovery states.
No additional training, tuning or diagnostic inference authorized by this report.
