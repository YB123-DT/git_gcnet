# Verified cross-dataset frozen Nested launch

Formal training is running, not completed. No final performance claim yet.

- Server: `ssh biggpu`; cmerc-repro `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Source: `2e3e710e54cdd85f9e660470a7d1ef1b3f93462e`.
- Immutable overlay:
  `/data2/yb/remote_experiments/osram_frozen_nested_cross_20261025/source_2e3e710`.
- Numerical source: e0ca94c for MOSEI/MELD;21e46f2 for IEMOCAP-6, as in README.
- Output: `/data2/yb/remote_experiments/osram_frozen_nested_cross_20261025/execution_v1`.
- Controller log: sibling `controller_v1.log`.
- tmux: `frozen_nested_cross_20261025`; controllerPID946470.
- Started:2026-10-25T22:41:56.852362UTC.
-2026-10-25T22:43:27UTC verified exact live commands for three initial jobs:
  MOSEI PID947598,MELD PID947730,IEMOCAP fold1 PID948036.
  Three running,53pending; subsequent automatic admission is expected.

## Scope and unchanged protocol

Seed66 only: MOSEI8rates×60epochs,MELD8×50,IEMOCAP-6 fivefold×8×100;
56formal own-rate frozen-parent/block pairs. No original Flat retraining.
Original Nested159235parameters; stage2constantLR3e-4,AdamWD1e-5,batch32,
cyclicGCNet-random-mask-v1; original task MSE/CE. Keep original LocalSkip,
CHM joint write/query/read-before-write and parent state fixed.

Current-rate Test-bestWF1 for MOSEI,Test-bestACC for classification; corresponding
UA/WF1 come from the same saved prediction/mask. INTERNAL TEST-ORACLE SCREEN;
not independent generalization or one deployed checkpoint.

## Checks before formal admission

28focused checks passed both locally and on remote cmerc-repro. All56completed
Signature references and protected parent checkpoints exist. Independently
validated saved own-rate arrays/metrics for one selection per dataset. Spec and
code-quality review passed,including actual-cycle concurrency gating. Compileall
and whitespace checks passed. No original runtime module/sampler was modified.

Remote test development initially lacked an already-committed shared test-only
fixture `tests/test_recent_noarea_serial_stage2.py`; copying that fixture into
the test overlay resolved collection. Final archive includes it; runtime code,
numerical source and data were not altered to make the tests pass.

Real two-epoch/one-batch-per-epoch GPU smokes all passed:

| Dataset | Seconds | Peak allocated MiB | Peak RSS GiB |
|---|---:|---:|---:|
| MOSEI |43.609143|447.838379|3.812782|
| MELD |16.792608|291.649414|2.633221|
| IEMOCAP-6 fold5 |14.544230|1245.101074|2.638458|

Each passed exact original selected-mask/logit replay and initial new-output
identity; finite connected gradients and nonzero actual Nested updates; original
parent unchanged/eval/no gradient; only Nested optimizer; original training
sample/mask/rate/update traces; one real CHM scan per batch; selected checkpoint
restore and all saved arrays replay. Smoke metrics are not formal results.

Torch1.13.1+cu117,Python3.10.13,NumPy1.26.4,sklearn1.3.2. The inherited optional
torch-sparse/libcusparse warning remains; native-tensor Nested does not use that
extension. No environment/dependency changes were made.

## Capacity and persistence

PhysicalGPU0/1/2 only,UUID-bound,maximum4taskworkers/card and12total. First admit
one formal job per dataset. The short smoke does not measure worst-case backward
memory; expand only after actual formal history covers all8rates,using observed
formal peaks plus reserve. Unrelated processes/GPU4 are untouched.

Full immutable block/Adam/RNG/epoch/history checkpoints and protected parent
references are retained. Compatible explicit resume adopts live workers without
duplicates. Errors stop further admission and remain visible,without killing
other jobs. The controller automatically fills the finite queue and produces
validated RESULT.md/JSON/CSV after all56complete. IEMOCAP equal-fold means and
pooled OOF are separate. GitHub receives only source/tests/small records.
