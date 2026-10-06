# Old Nested cross-dataset screening

INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT.

User requested Nested on MOSEI and IEMOCAP. Use the original
`nested_gnn_rooted_evidence` (Local single node, eight Memory head nodes per
active evidence, 64d, three GIN layers), not Local8/root-aware variants.
No changes to model code, Memory read/write/query, original task head/loss,
random cyclic missing protocol, optimizer or dataset-specific preprocessing.

## Fixed scope and protocol

- MOSEI: seed66, official single split,100epochs, missing rates0.0–0.7.
- IEMOCAP: six classes, seed66, five leave-one-session-out folds,100epochs
  each, same eight missing rates. No three-seed expansion.
- New configs copy each exact dataset/fold Flat config; only
  `osram_meaningful_block` changes. Null rate list in historical config uses
  the trainer's existing default eight rates, not a new sampling schedule.
- Per-rate BEST Test-oracle screening. Current trainer selects MOSEI by
  weighted F1 and IEMOCAP by accuracy; report IEMOCAP W-F1 and accuracy.
- Existing Flat metrics do not record selection_metric. Do NOT label them
  as proven identical checkpoint-selection controls. Reuse them for masks,
  inputs and protocol audit, and flag any historical numerical comparison.
- Compare ordered and canonical evaluation masks on completion; retain all
  eight BEST checkpoints, predictions and complete last_training.pt state.
- Server biggpu. GPU5 dedicated to MOSEI; GPU6 runs IEMOCAP folds sequentially.
  GPU4 forbidden. Capacity checks do not alter batch size or training protocol.
- Source pinned via git archive and SNAPSHOT.json; INPUTS.json hashes feature
  files, label/split pickle, exact reference configs and metrics.
- Failed child stops that lane; no blind rerun or silent weights-only resume.

## Commands

From sealed source, using `/data2/yb/reproduction_workspace/envs/s0/bin/python`:

```text
python -m experiments.osram_nested_cross_dataset_20261006.run --prepare
python -u -m experiments.osram_nested_cross_dataset_20261006.run --lane mosei --gpu 5
python -u -m experiments.osram_nested_cross_dataset_20261006.run --lane iemocap --gpu 6
```

Root: `/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006`.
Persistent tmux queues record PID, command, UUID, log and output paths.

## Verification and status

Three focused runner tests: exact six-task scope, only old-Nested config
switch changes, invalid protocol/fold/rates/interventions rejected.
Tests first failed on absent runner, then passed. No repeated model smoke:
the unchanged old Nested model already has regression-test evidence.

## Launch snapshot

Source commit33ded09 (pushed), sealed archive:
`/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/source_33ded09`.
INPUTS.json prepared successfully with both datasets and six reference folds.
Remote s0 environment also passed all three runner tests.
Started2026-10-06T01:31:36 UTC:

| Lane | Host GPU | Queue PID | First child PID | tmux session |
| --- | ---: | ---: | ---: | --- |
|MOSEI|5|3001323|3001353|nested_mosei_20261006|
|IEMOCAP six-class|6|3001326|3001355|nested_iemocap_20261006|

LAUNCH_MOSEI.json and LAUNCH_IEMOCAP.json preserve exact commands/UUIDs/paths.
Live checks confirm both child PIDs; IEMOCAP logs show feature loading.
This records successful process launch, not epoch completion or final scores.
IEMOCAP remaining folds are queued in the same persistent process.

Status: running (initial data loading). No new scores available.
