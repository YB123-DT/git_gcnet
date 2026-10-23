# Launch and verification handoff

INTERNAL TEST-ORACLE SCREEN. Active v2 completed150epochs. The launch notes
below are retained as historical provenance, not current running status.

## Completion verified2026-10-23 10:10 UTC

`execution_v2/SUMMARY.json` is complete at150epochs,1632.720seconds(27.21minutes).
All8 checkpoint/prediction SHA256 pairs verified; independent sklearn recomputed
selected W-F1 from686valid targets per rate. Labels,speaker,utterance IDs/order
and exact availability match s0. All150 training mask traces,update counts and
LRs match. Each selected epoch is the earliest strict TEST-best W-F1. Initial
weights are byte-identical. No new inference or training was performed to report.

| Environment | Mean ACC | Mean W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| s0 |79.2683|79.2491|73.3740|73.4098|
| cmerc-repro |79.5351|79.4604|73.4756|73.3516|

Mean W-F1+0.2114pp,high W-F1-0.0582pp. This single seed does not establish a
stable environment advantage; mean MSE1.496185 vs s0's1.494741 is slightly worse.
Small original RESULT.md/JSON are archived in `completed_v2/`; all weights and
large predictions remain remote. No additional seeds/modules are launched.

## Retained launch identity

- Server: `ssh biggpu`; physical GPU7,
  `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`; process-local cuda:0.
- Environment: `/data2/yb/conda_envs/cmerc-repro/bin/python`, Python3.10.13,
  Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2.
- Numerical source: `5877678ce3fac9358a94f1217e3dbac5509f6f58`, exactly the
  accepted s0 reference; independently overlaid runner revision `1d5d5ec`.
- Immutable source:
  `/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/source_5877678_runner_1d5d5ec`.
- Active output:
  `/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2`.
- Persistent tmux: `mosi_cmerc_v2_20261023`; worker PID2260273.
- Log: `/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/train_v2.log`.
- One seed66 original Flat, no Nested/V3,150epochs,3e-4 for100 then10x lower
  for50; original task,MSE,batch32,Adam,WD1e-5,new GCNet missing and Test-best
  W-F1 unchanged. No other environment/dataset experiment is authorized here.

## Verified

- Four regression tests pass locally and in remote cmerc-repro; covers rejecting
  s0, rejecting numerical source drift, training mask/update/LR parity without
  falsely requiring identical cross-environment loss, exact final LR expression.
- All25 baseline numerical/setup source hashes match; no model source edited.
- Initial Flat parameters are byte-identical to s0:
  `89718ab6425657b9a05286892ff7f8d93bca0cac5891f3f0d5e601e49af26ef2`.
- Original13,509,793 parameters; data manifest and all train/dev/test splits
  verified; exact saved evaluation banks loaded, not independently resampled.
- Effective config SHA256:
  `5d6e22c06fbf02ff94423670546cfcc3d3738f7d8aefdaa12724693d27fb8d73`.
- Active v2 verified through epoch2/150,43.20seconds,peak1223.14MiB; first-epoch
  LR3e-4,two optimizer updates,72connected/nonzero gradient tensors,training mask
  trace `658cc5e40a18f2306f38bf5a9e226687d32d8b08bc9058e4a53c636c5a5cdfb4`.
  Training mask/update/LR traces match s0. Initial v1 also verified eight
  complete training epochs with matching training mask/update/LR traces and
  finite gradients to all72 connected parameter tensors. This is implementation
  verification only, not performance evidence.
- Existing s0 MOSEI process and other users' jobs left untouched. GPU4 not used.

## Superseded launch, not a result

`execution_v1` / runner `c3b2fad` was interrupted deliberately after8 fully
recorded epochs, before the epoch101 LR boundary. The final LR literal was
replaced by the exact reference expression `3e-4 / 10`, avoiding strict float
comparison failure at epoch101. The active v2 starts fresh with corrected runner;
it is not described as a full resume. v1 source,logs,masks,predictions,checkpoint
files are retained unchanged and excluded from performance comparison. It may
have a partially interrupted atomic checkpoint transaction; never select it as a
completed run. No old/new-missing checkpoint was deleted.

## Original pending-work note, now completed

Active training must reach150epochs and complete the selected-artifact integrity
checks before reporting final scores. It automatically emits RESULT.md/JSON
using the exact saved selected predictions, with s0 vs cmerc per-rate/eight-rate/
high-missing ACC,W-F1,MSE. No fresh randomized inference or automatic extra seeds.
After completion, fetch only small records and publish them; checkpoints stay
remote. A change between these software environments does not identify which
package caused a difference, and one seed is not a multi-seed conclusion.

The existing optional PyG torch-sparse import warning is retained in the log;
the Flat's actual GPU forward/backward succeeds without that extension. No
dependencies were installed, upgraded or changed.
