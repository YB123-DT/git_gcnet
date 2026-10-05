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

Status: prepared, launch verification pending. Multiseed results pending.
