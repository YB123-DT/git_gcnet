# MELD first-round Memory write tuning

INTERNAL TEST-ORACLE SCREEN. User authorized first-round launch2026-10-17.
Exactly five from-scratch seed66/50-epoch jobs; no later round or Optuna.

| ID | write_step | write_ridge |
|---|---:|---:|
|W0|0.6|0.001|
|W1|0.3|0.001|
|W2|0.9|0.001|
|W3|0.6|0.0001|
|W4|0.6|0.01|

All other settings inherit the accepted ECERC Flat1600/Local-Skip baseline:
batch32,Adam lr0.001,weight_decay1e-5,dropout0.5,gcnet-random-mask-v1,
cyclic0.0–0.7,emotion-only/no-JEPA,8heads,K/V64,one joint observed-A/T/V write.
Read ridge and all Local/Base/Gap/query/head definitions are unchanged.

Numerical training/loader source remains the clean accepted worktree at
b4a313841f098487a6482d2712bdac94aa8e319e,not the newer training implementation.
The driver only wraps existing evaluations and complete-state checkpoints.
Existing Test-best ACC per-rate selection is retained and clearly exploratory.
UA/W-F1/CE/support/recall come from the exact same selected epoch/logits/mask.
Historical trainer final resampled evaluation is retained separately,not used
to fill BEST scores. There is no added evaluation or mask-sampler call.

## Resource and execution

Server:ssh biggpu. Environment:/data2/yb/conda_envs/cmerc-repro/bin/python.
Use healthy host GPUs5/0,UUID-pinned;each at most4tasks,total at most5.
GPU4is forbidden;no unrelated process is stopped. W0's first actual training
epoch provides the real forward/backward/optimizer/peak-memory admission check.
Only after it commits does the queue admit W1–W4 using measured peak plus25%
and a4GiB minimum reservation. Batch or model size are never reduced to fit.

```bash
cd IMMUTABLE_DRIVER_WORKTREE
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_write_tuning_20261017/run.py \
  --model-source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138 \
  --output /data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1
```

Run in tmux/persistent runner. Actual source SHA,PID,GPU UUID,config hashes,
dataset SHA and paths are recorded in RUNNER/QUEUE/SUMMARY JSONs and LAUNCH.md.
Output preflight requires12GiB disk;each child independently verifies data and
source. No environment changes or dependencies are installed. The process-local
Torch1.13 UUID enumeration shim corrects both device-count aliases before CUDA
seeding and recovery;all five jobs use the same shim.

## Saved evidence and bounded recovery

- INITIAL_STATE.json:exact pretraining model state hash and parameter count.
  Every variant must match this run's W0 before its first optimizer step.
- MASK_TRACE.json:actual padded effective availability,speaker and ID hashes;
  per-epoch training and all400selection evaluations,with missing fractions and
  pattern counts. No alternative mask schedule is substituted.
- Eight best_miss_*.pt plus exact selected NPZs tied by checksum in each BEST.
  Selection artifacts include logits,labels,predictions,IDs,speakers,availability.
- last_training.pt:original full model/optimizer/RNG/epoch/schedule recovery.
  Trace recovery trims uncommitted rows and restores selection epoch counters.
- EPOCH_STATUS.json:committed epoch,selected-score snapshot,trained alpha/beta,
  measured CUDA allocation peak. It is progress,not final result confirmation.
- SUMMARY.json:completed all-eight-rate ACC/UA/W-F1/CE and high-missing mean.
- RESULTS.json:automatic five-run summary and exact initial-state/mask-stream
  equality checks. Failed/incomplete runs cannot be promoted as completed.

Only unreferenced temporary model/selection metadata in THIS NEW run's
training_versions cache are compacted using the existing protected-reference
pattern. Every current per-rate BEST and full last_training state is retained;
protected historical24checkpoints are never targets. Selected NPZs are retained.
The queue permits at most one audited OOM retry with the identical config;
full recovery requires an actually committed complete state,never just BEST.
An interrupted first epoch without a complete state fails loudly instead of
claiming exact resume. Restarting the controller into an existing output is
intentionally rejected;operator recovery must inspect the existing queue first.

When all jobs finish,the controller checks initialization and actual training/
selection mask traces against W0. Only these paired checks justify treating the
five jobs as a controlled single-factor comparison. Different selected epochs
can still have different selected test masks;the full per-epoch streams and
selection metadata are retained rather than pretending they are identical.

Tests:tests/test_meld_write_tuning_run.py and test_meld_write_tuning_capture.py.
They check finite configuration scope,one-factor differences,admission gating,
UUID commands,seven-class metrics,exact sample order,no extra forward/random
draws,final-vs-selected distinction,hook restoration,and trace recovery.
Tests do not establish trained performance or an optimal MELD setting.
