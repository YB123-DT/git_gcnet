# Verified launch: one MELD Flat warmup/cosine run

INTERNAL TEST-ORACLE SCREEN. Running, not a completed performance result.

- Started2026-10-17T12:52:57UTC on biggpu (host user22).
- Immutable driver `ae142247ac5be7b56f7ad532f0f26709f7a893d5`.
- Immutable numerical source `b4a313841f098487a6482d2712bdac94aa8e319e`.
- Driver checkout: `/data2/yb/remote_experiments/osram_meld_cosine_20261017/driver_v1`.
- Output: `/data2/yb/remote_experiments/osram_meld_cosine_20261017/runs_v1/C0`.
- tmux session `meld_cosine_v1`; controller PID3122793, worker PID3123068.
- Physical GPU5 UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`, logical cuda:0.
- Controller log: `/data2/yb/remote_experiments/osram_meld_cosine_20261017/controller_v1.log`.
- Training log: `runs_v1/queue/logs/C0.attempt1.log` under the remote experiment root.
- Python `/data2/yb/conda_envs/cmerc-repro/bin/python`:3.10.13,
  Torch1.13.1+cu117, NumPy1.26.4. No environment changes.
- Existing numerical files, W0, protected historical weights and other tasks
  were not modified or stopped. One additional training job, no search queue.

## First committed epoch

`LAUNCH_STATUS.json` is an immutable epoch1 snapshot, not live status.

- Real train/evaluate/backward/optimizer and epoch save completed.
- Actual first epoch optimizer group LRs: `[.0005,.0005,.0005]`.
- Initialization hash equals W0:
  `c2ea3d6fdf2d6e559283a5bd15e4e2b3bea239b2e95aba161f5a271102402008`.
- Parameter count13,290,795, unchanged from W0.
- Actual training mask hash and all eight test selection-mask hashes equal
  W0 epoch1; no added sampler calls or inference passes.
- Torch peak allocated559.565MiB (not total nvidia-smi context memory).
- ACC-best and UA-best each retained eight checkpoints and same-forward
  prediction artifacts; each evaluation contains2610 valid test utterances.
- A real checkpoint audit recomputed ACC/UA/WF1/CE from saved artifacts for
  rates0.0/0.7 in both selectors, verified hashes and selected-epoch parity.
- `last_training.pt` contains model, optimizer, RNG, schedule progress and
  eight UA selection references plus LR/metric histories at its committed
  epoch. Canonical BEST files may advance during an uncommitted next epoch;
  recovery repairs them from the last committed hashed references.
- No worker failures at verification. The pre-existing optional torch-sparse
  import warning remains; the OSRAM training path ran successfully without it.

## Verification and limits

22 tests passed locally and in the actual remote cmerc-repro environment.
Compileall and staged whitespace checks passed. This verifies execution and
recording, not performance improvement or statistical significance.

Only the existing schedule switch changes. The .05 warmup ratio activates two
warmup epochs at50 total epochs, then cosine reaches0 at epoch50.
The matched control is W0 ACC-best60.838123/UA35.632827, not the historical
three-seed mean. UA-best is a separate test-selected exploratory table; W0
did not retain UA-best and cannot silently be relabeled as that comparator.

The persistent controller will stop after this single50-epoch job and save
`RESULTS.json`. No automatic extra seeds or methods are authorized here.
