# Stage 2: frozen Flat, train Evidence Gate only

User authorized 24 paired tasks: seeds66/67/68 times target rates0.0–0.7.
Each task loads its original `best_miss_0pX.pt` from
`/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_SEED`.
These checkpoints were selected by Test-oracle, not validation. All results
remain internal diagnostics and cannot be reported as unbiased paper results.

## Fixed design

- No Stage1 retraining. Stage2 starts a fresh optimizer, not a resumed run.
- Freeze every original parameter (ObservedSetEncoder, OSRAM including Local,
  emotion_adapter/local_skip, classifier and auxiliary modules).
- Keep original modules in eval mode even when the training loop calls train;
  only `osram.local_evidence_gate` is trainable/in train mode.
- Do not put the downstream Flat/classifier under no_grad: gradients must
  propagate through their fixed operations to the Gate.
- Four evidence-specific scalar gates, gamma0.2; emotion loss +0.001 times
  active-valid mean squared deviation from1. No new module or extra objective.
- Keep original cyclic random-missing training, batch32, original Adam settings,
  100 Stage2 epochs. Target rate identifies the parent checkpoint/evaluation,
  not a restriction of the training mask schedule.
- Save initial epoch0 evaluation, best Gate, last Gate/optimizer/progress/RNG,
  history, final-epoch score, source checkpoint hash and frozen-state hashes.
- Reproduce parent score at epoch0; assert nongate parameters AND buffers stay
  unchanged at every epoch and before any best restore.

Epoch0 is included in the Test-oracle best candidates. Thus a nonnegative best
delta is guaranteed by retaining the unchanged parent; it alone does not prove
learned gating works. Report selected epoch, final score and whether any trained
epoch beats the parent, not only the nondecreasing best score.

## Server and scope

Only biggpu host GPU0 (`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`), max3
concurrent jobs with resource checks; GPU4 prohibited. Independent output per
seed/rate; no overwriting previous experiments. Source weights stay untouched.
The existing joint-training experiment remains intact as a distinct comparison.

## Implementation plan

1. Tests first: strict source loading, train-mode lock, gate-only optimizer and
   finite nonzero gate updates with unchanged original state.
2. New standalone runner reuses existing train_epoch/evaluate_rate; no change to
   normal training behavior in the core model/trainer.
3. Remote one-epoch smoke on seed66/rate0.7; check epoch0 reference reproduction,
   frozen hash, saved Gate artifacts and finite updates.
4. Commit source, isolate snapshot, launch24 tasks; verify actual child progress
   and archive launch/provenance. Push scoped changes automatically.

## Commands

From the isolated remote code directory using the existing s0 Python:

```sh
python -u experiments/osram_frozen_evidence_gate_20260930/run.py --launch --max-tasks-per-gpu 3
```

Default outputs: `/data1/yb/remote_experiments/osram_frozen_evidence_gate_20260930/runs`.
The launcher owns a persistent lock and detached coordinator; do not rerun it
against the same directory. Task order is seed then target rate. On child failure,
stop admission and record remaining tasks, while existing children drain.
`best.pt` and `last.pt` contain Gate weights, optimizer and RNG plus source hash;
they require the referenced immutable Flat checkpoint. Automatic resume is not
implemented, and this task is not reported as a resumed run.
