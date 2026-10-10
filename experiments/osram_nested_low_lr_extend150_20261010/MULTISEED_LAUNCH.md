# Nested lower-LR continuation: seeds67/68

INTERNAL DIAGNOSTIC ONLY; cumulative per-rate Test-oracle selection.

User authorized Nested three-seed comparison after completed seed66 continuation.
Seed66 is not rerun. Source states for67/68 are complete constant100 runs under
`/data2/yb/remote_experiments/osram_readout_top3_3seed_20261005/runs/nested_gnn_rooted_evidence/seed_{67,68}`.
Continue epochs101–150 at constant1e-4; model, Adam moments/steps, RNG and existing
selection references restored. Only epochs and learning_rate change.

Server biggpu physicalGPU7, UUID `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`.
Prelaunch free memory32,491MiB. Two persistent parallel tmux sessions:

| Seed | PID | Session | Output |
|---|---:|---|---|
|67|2947407|lowlr_nested_67|runs/nested_seed67|
|68|2947493|lowlr_nested_68|runs/nested_seed68|

Root `/data1/yb/remote_experiments/osram_nested_low_lr_extend150_20261010`;
wrapper `source_multiseed/run.py`, code5781797, committed/pushed before launch.
Arguments match seed66 LAUNCH.md except original source/output/seed and wrapper
path/commit. Logs `logs/nested_seed{67,68}.log`. Original model source_ad211c0
remains checksum-verified and unchanged.

Both actual epoch101 traces verified optimizer step200 and all three groupLRs
1e-4 before any new update. Both PIDs alive on correct GPU; seed67 reached102
and seed68 reached101 at verification. Performance results pending.

First launch stopped before state copying/training due solely to tuple versus
JSON-list configuration representation. Equality now compares canonical JSON,
still rejects changed values; four regression tests pass. Failed logs and source
preserved as `*_attempt1_failed.log` and `run_attempt1_failed.py` on remote.
No original checkpoint was modified. No training was duplicated.

Completed seed66: cumulative8-rate81.162868%, high-missing76.077229%.
Original100:80.992147%,76.077229%. Only rate0.3 refreshed; report cumulative and
101–150-only results separately, not inherited BEST as new-epoch performance.

Flat67/68 have no complete recovery state; do not create weight-only restarts or
new100epoch Flat runs as equivalent paired continuation without authorization.
