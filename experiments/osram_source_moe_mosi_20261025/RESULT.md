# MOSI 12-slot Source–Role MoE

INTERNAL TEST-ORACLE SCREEN

Status: both formal 150-epoch runs started and verified running on biggpu GPU0.
Last launch check: residual epoch4, replace epoch3; eight selected-rate artifacts
exist for each run, training mask/update/LR parity passes and no error is recorded.
These are early progress observations, not final performance results.

Source commit: `b92490d` (uploaded to `github/feature/ecerc-features`).
Immutable source: `/data2/yb/remote_experiments/osram_source_moe_mosi_20261025/source_b92490d`.
Runs: `execution_v1/residual` and `execution_v1/replace` below the same parent.
Persistent tmux sessions: `source_moe_residual_20261025` (PID3505754),
`source_moe_replace_20261025` (PID3505822). No other task was modified or stopped.

| Arm | Allocated parameters | Trainable parameters | MoE parameters | GPU smoke peak MiB | Smoke seconds |
|---|---:|---:|---:|---:|---:|
| Flat + MoE residual | 14,897,331 | 14,897,331 | 1,387,538 | 3112.79 | 18.41 |
| MoE replaces Adapter | 14,897,331 | 5,362,227 | 1,387,538 | 3290.44 | 15.15 |

The replacement retains 9,535,104 frozen/unused original Adapter parameters for
initialization/state-dictionary compatibility; they do not participate in prediction.
The two connections do not have matched active capacity. Each expert independently
maps all 12 unpooled slots plus Local and metadata; no expert is assigned a modal role.

## Verification

- 7 core tests pass locally in comp-repro, including exact pre-change default-off
  train outputs/gradients/RNG against 6400940; this environment is **not** used to train.
- Remote cmerc-repro: 7 tests pass (module/runner); historical Git-source regression
  is locally verified because the remote preflight snapshot has no Git object database.
- Real MOSI smoke for each arm: three epochs, one scan per batch, original training
  mask hashes/update counts/LRs exact; all 38 added parameter tensors receive finite,
  nonzero gradients after opening the zero-initialized output.
- Initial predictions exactly match original Flat on 686 test utterances at each
  of rates 0.0 and 0.7, before training. Strict checkpoint reload predictions match;
  model/Adam/config/CPU and UUID-bound CUDA RNG states are saved.
- Host biggpu GPU0 UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45; cmerc-repro
  Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2. GPU4 never used.
- First preflight failed before data loading because config.py resolved the label
  path before the manifest environment variable. Runner now sets it before imports.
  Failed preflight artifacts retained. No features, masks or model equations changed.
- Existing optional torch-sparse loader warning appears (libcusparse.so.11 missing);
  this unused optional backend did not prevent either real forward/backward/step.

Smoke is not evidence of improved performance. Router diagnostics are explicitly
last-training-batch statistics, not complete-test-set attention/usage statistics.

## Evaluation protocol

MOSI66, new gcnet-random-mask-v1, original cyclic eight rates, batch32, Adam,
WD1e-5, MSE, LR3e-4 epochs1–100 and 3e-5 epochs101–150. No auxiliary objective.
Reuse accepted cmerc Flat exact test availability banks, ordered labels and speakers.
Per-rate Test-best W-F1 is exploratory; it is not one deployed checkpoint's score.
Accepted reference: eight-rate ACC79.535061/W-F179.460447, high W-F173.351571.

Replacement bypasses Adapter Dropout, so cross-arm training CUDA random draws
are not identical, despite identical data order, missing masks and module initialization.
Extra source decomposition tracks joint-write Value contributions, not pure modality causes.

Runtime outputs: `/data2/yb/remote_experiments/osram_source_moe_mosi_20261025/`.
Each completed run publishes per-rate ACC/W-F1/MSE and selected epochs in its
`RESULT.md`, `SUMMARY.json`, `best_by_rate.json`, with protected BEST/last weights
and same-call predictions. Final scores are pending completion.
