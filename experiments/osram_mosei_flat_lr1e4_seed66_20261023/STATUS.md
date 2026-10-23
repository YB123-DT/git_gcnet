# Seed66 completion: running, not a final score

2026-10-23 08:08 UTC. INTERNAL TEST-ORACLE SCREEN.

The one authorized MOSEI Flat seed66 constant1e-4 job is actually running on
biggpu physicalGPU7. No Nested/V3 experiment was launched. Existing67/68 runs
and every old checkpoint/prediction/mask remain unchanged.

- Immutable source `015588dd37669fdad43e1ee9cd0e7464a0ca0286`.
- Controller PID2119471; worker PID2119641.
- tmux `mosei_flat_lr1e4_seed66_20261023`.
- Output `/data2/yb/remote_experiments/osram_mosei_flat_lr1e4_seed66_20261023/execution_v1`.
- Worker log `train_seed66.log`; live run `runs/CMUMOSEI/seed_66/flat/SUMMARY.json`.
- Config: original13,509,793-parameter Flat,60epochs,batch32,Adam,WD1e-5,
  constant1e-4,MSE,GCNet-random-mask-v1/cyclic0–.7,per-rate TEST-best W-F1.
- Environment unchanged: s0 Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2.

## Actual first-epoch verification

Epoch1 completed:71 updates on16,326 valid training utterances; actualLR[0.0001];
taskMSE1.6644473421;72 backbone parameters had connected finite nonzero gradients.
The worker's same-seed mask-trace/update assertions passed:
`68a0fc8c70e4ca2589ba98a2bc806adbf62b1aa33bbe670766d0d853c59e9946`.
Initialization, official data/splits and all test-mask hashes matched the retained
same-seed protocol reference. Full effective config/launch identity: LAUNCH.json.
These are execution checks, not evidence of final performance or improvement.

## Tests

RED: extending the existing real-config test to seed66 failed against immutable
old worker0da38b3 specifically because it rejected66; the other4 cases passed.
GREEN: current source remote suite passed13 checks: original67/68 config/queue
regressions plus66, bounded completion command and reused schema-normalization
tests. Only the pre-existing PyG distributed deprecation warning appeared.
Local8 lightweight command/config checks, compile and diff checks also passed.
No redundant standalone training smoke; original Flat numerical/GPU path had
already been verified, and the formal first epoch was checked directly.

After60 epochs the controller automatically verifies selected checkpoint and
prediction hashes and reports the new66 plus retained67/68. It does not infer
success from process exit alone and does not rank unfinished runs. Results will
be RESULT.md/RESULT.json/PER_RATE.csv. No further training/HPO is queued.

The source extension changes only seed admission and default-schema validation;
the old67/68 launcher remains exactly two jobs. No model, Memory, optimizer,
task loss or missing sampler computation was modified. Training is unfinished.
