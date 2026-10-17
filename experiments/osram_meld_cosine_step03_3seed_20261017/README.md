# Finite MELD cosine three-seed replication and step0.3 comparison

INTERNAL TEST-ORACLE SCREEN. Five authorized new runs, then stop.

## Grid

| Job | Seed | Schedule | Write step | Status at plan |
|---|---:|---|---:|---|
| Reused C0 |66|existing warmup+cosine|0.6|completed; do not retrain|
| C67 |67|same|0.6|new, phase1|
| C68 |68|same|0.6|new, phase1|
| S66 |66|same|0.3|new, phase2|
| S67 |67|same|0.3|new, phase2|
| S68 |68|same|0.3|new, phase2|

Phase1 runs C67/C68 concurrently. When both finish, the finite controller
automatically admits S66/S67/S68 concurrently, subject to actual GPU/disk
admission. Maximum4 of this task's processes on physical GPU5; actual maxima
are2 then3. A failed phase1 prevents unsupported phase2 comparisons; failures
are recorded rather than hidden or retried with different hyperparameters.

The seed66 reference is the already completed C0 at
`/data2/yb/remote_experiments/osram_meld_cosine_20261017/runs_v1/C0`:
ACC60.924330/UA36.642175/WF1 57.349488, ACC-best eight-rate means.
Its original weights, selected predictions and full source provenance are
reused read-only, not copied into a pretend new training run.

## Fixed protocol

Same biggpu server and `/data2/yb/conda_envs/cmerc-repro/bin/python` environment;
immutable numerical source `b4a313841f098487a6482d2712bdac94aa8e319e`.
Actual model, original GCNet sampler and task training are unchanged.

- ECERC feature SHA256:
  `a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.
- MELD, official splits,50epochs, batch32, Adam peakLR.001, weight decay1e-5,
  dropout.5, warmup_ratio.05 (two warmup epochs), then existing cosine to0.
- Original gcnet-random-mask-v1, cyclic0.0–0.7, no-JEPA/emotion-only.
- Flat1600, original Local Skip;8heads, K/V64; causal read-before-write and
  one joint observed-modality Block Delta Write. Ridge remains.001.
- No Block2/Nested/Gate, no new loss, no persistent mix or completion.
- Primary comparison: per-rate Test ACC-best, UA/WF1 from the exact same
  selected forward. Separate UA-best selector and all-epoch metrics retained;
  never mix ACC-best ACC with UA-best UA.

## Source and pairing

The new launcher reuses the verified cosine driver and original captured
numerical worker. A backwards-compatible `paired_control` flag only changes
which observational evidence is checked: defaultTrue preserves prior behavior.
C67/C68 have no freshly trained same-seed constant-LR control and explicitly
record null for paired initialization/mask equality; they must not pretend to
match W0 seed66. Each still records the actual initialization and mask stream.

S66/S67/S68 compare initialization and all actual train/selection mask hashes
with C0/C67/C68 respectively, at each epoch. Only write_step differs within
each pair. This tests the combination, not an assumed additive gain from the
earlier constant-LR step0.3 result.

Changing a seed does not silently change any other configuration. Config
tests verify the entire dictionary. Source/data hashes and effective config
are saved per run. Numerical source and old result directories are not edited.

## Output and resources

`/data2` has only approximately12GiB free, so ONLY new artifacts use the same
server's `/data1` partition (approximately39GiB free at preflight):

`/data1/yb/remote_experiments/osram_meld_cosine_step03_3seed_20261017/runs_v1`.

This is not a server/environment/data migration. No old checkpoints are deleted.
Physical GPU5 remains UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`;
the process sees cuda:0. GPU4 remains forbidden. Use measured previous
Flat peak564.897MiB, conservatively reserve4GiB GPU and3GiB disk per active job,
plus margins. Do not reduce batch or algorithm dimensions to admit jobs.

Retain eight ACC-best and eight UA-best weights per new job, exact labels/
predictions/masks/hashes, all-epoch metrics, LR trace and complete optimizer/
RNG/selection recovery state. Queue supports the existing bounded OOM retry
and exact epoch recovery. Do not call loading BEST an optimizer continuation.
`--resume` requires the same immutable code/config/queue.

## Launch command from immutable driver root

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
 experiments/osram_meld_cosine_step03_3seed_20261017/run.py \
 --model-source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138 \
 --reused /data2/yb/remote_experiments/osram_meld_cosine_20261017/runs_v1/C0 \
 --output /data1/yb/remote_experiments/osram_meld_cosine_step03_3seed_20261017/runs_v1
```

Use tmux for persistence. RESULTS.json will contain both three-seed groups,
per-seed/per-rate scores and mean/sample-SD summaries for ACC-best and UA-best
separately. It is written after the finite queue ends, including missing/failed
runs. No new datasets, HPO, seeds or methods are automatically authorized.

## Checks

Tests first failed on absent finite launcher / optional observer mode, then
check five exact configs, seed66 reuse, correct same-seed controls, phase
dependency/failure propagation, GPU5 UUID commands and unchanged default
observer behavior. Reused capture, LR, RNG and ACC/UA selection tests also run.
The actual first optimizer epoch of C67/C68 is the new-seed GPU integration
check, not an extra smoke training condition or performance evidence.
