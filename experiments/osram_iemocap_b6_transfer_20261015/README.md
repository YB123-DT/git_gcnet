# IEMOCAP B6 transfer under the original Flat protocol

This finite experiment transfers only the selected B6 PCA settings to
IEMOCAPFour and IEMOCAPSix. It schedules seeds 66/67/68 and folds 5,1,2,3,4:
30 fold runs total. The first six queued jobs are fold5. Fold5 results are
reported separately; they are never called a fivefold average.

Each run starts from its own original Flat `config.json` in:

```
/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/iemocap{4,6}/seed_{66,67,68}/fold_{1,2,3,4,5}/config.json
```

The launcher validates 100 epochs, batch32, Adam, learning rate .001,
weight decay 1e-5, dropout .5, original 1600-dimensional Flat readout and
Local Skip, emotion-only training, cyclic rates 0.0–0.7, GCNet random masks,
and per-rate Test-best ACC selection. Other original fields, including any
feature paths, remain as recorded in the source configuration. The old
omitted/dataset-default selection metric means accuracy for IEMOCAP.

The only model changes are:

| Setting | Value |
|---|---|
| `osram_meaningful_block` | `learned_robust_pca_evidence_decomposition` |
| `osram_rpca_rank` | 2 |
| `osram_rpca_steps` | 14 |
| `osram_rpca_threshold_init` | .33599569865951234 |
| `osram_rpca_lr_multiplier` | 2.8200380294401013 |

Bookkeeping explicitly enables selection artifacts, records accuracy
selection, and selects logical `cuda:0` after UUID pinning. MELD's tuned
global learning rate/dropout/weight decay are not imported. B6 settings were
selected on MELD; this transfer remains an internal test-oracle diagnostic.

## Standalone deployment

Keep launcher and worker outside the immutable numerical checkout at
`/home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012`.
That checkout must remain at commit
`785a01242ef1a82b97d5337acbb33ba963df5537` without tracked edits. Only after
resolving this source does the launcher import its existing scheduler.

The worker CLI is:

```bash
python /absolute/deployment/worker.py \
  --source-root /home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012 \
  --config /absolute/effective-config.json \
  --output /absolute/unique-run-output \
  --gpu-uuid GPU-HEALTHY-UUID
```

Create the authorized real-batch smoke once with this same worker's `--smoke`
option and a planned effective configuration. The launcher consumes its
existing `SUMMARY.json`; it never reruns smoke. Require `status=complete`,
`smoke_only=true`, matching configuration, and a positive finite measured
`peak_memory_mib`. Use the longest IEMOCAPSix fold5 batch to bound both tasks.
Record feature/label paths and data checks in `DATA_PROVENANCE.json` before
launching; the launcher stores that JSON and its SHA256.

```bash
python /absolute/deployment/run.py \
  --worker /absolute/deployment/worker.py \
  --smoke-summary /absolute/smoke/SUMMARY.json \
  --data-provenance /absolute/DATA_PROVENANCE.json \
  --output /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015 \
  --check-only
```

Remove `--check-only` for the authorized durable queue. `--source-root` and
`--baseline-root` are available for explicitly locating the recorded inputs.
Use the server's established persistent process mechanism; preserve its PID,
command and log path in the project run record.

## Resources, recovery and artifacts

Only physical GPUs 1 and 3 are eligible. GPU1 permits at most eight total
actual compute processes, counting this queue's launch reservations once.
GPU3 permits at most two jobs belonging to this queue. GPUs 4 and 5 are
excluded. Each admission refreshes UUIDs, actual compute PIDs, free memory,
and disk space. Measured smoke peaks, a 1024 MiB reserve and complete peak
reservations for active jobs may reduce concurrency below those maxima.
No batch size or training protocol changes are made to increase capacity.

The existing scheduler supplies queue/controller locks, PID birth identity,
unique output checks, persisted launch commands, recovery and completed-run
reuse. A completed worker must have an exact-config, exact-dataset/seed/fold
summary with finite ACC/UA/W-F1/CE at all eight rates. Failed or malformed
results never enter aggregates. Its single OOM retry lowers capacity without
changing configuration; worker recovery must validate complete training state.
External processes are observed but never signalled or adopted.

`MANIFEST.json` records all 30 original configs, paths/hashes, exact config
diffs, effective configs, frozen source SHA, launcher/worker SHA256, data
provenance, smoke provenance and environment. Per-job launch provenance is
also saved in `queue/provenance/`; logs/configs/state remain under `queue/`.
Manifest changes are rejected on restart to prevent mixed experiment versions.

After each completed job, `SUMMARY.json`, `SUMMARY.csv`, `RESULT.md` and
`FOLD5.json` are refreshed. Equal-fold fivefold means require all five folds
for each seed and all three seeds for the dataset. Three-seed uncertainty is
sample standard deviation. Reports cover new B6 results only. Historical
exact-selection-epoch Flat UA is marked unavailable; old ACC or fresh final
reevaluation must remain separate evidence.

## CPU verification

```bash
python -m pytest -q tests/test_iemocap_b6_transfer_queue.py tests/test_meld_block2_queue.py
python -m py_compile experiments/osram_iemocap_b6_transfer_20261015/run.py
```

The queue tests cover immutable source configurations, bounded fold ordering,
GPU process/memory admission, exact-summary validation, separate fold5 and
fivefold reporting, complete-output adoption, 30 real CPU subprocesses, and
restart reuse. These checks do not execute GPU training.

## Verified deployment preparation

The numerical source is pinned to785a012, not a changed working tree. Compared
with9fa9f57, its only trainer change expands two MELD-only artifact guards to
IEMOCAP classification. It does not change any model, optimizer or sampler.
Tests compare captured versus uncaptured predictions, loss and both RNG states.

`GPU_SMOKE.json` is the completed67.11-second real IEMOCAPSix fold5 check:
batch32, padded110, three updates at missing rates .7/.7/.0, finite task loss
and gradients, B6 parameters actually updated. CUDA peak allocation3333.87MiB,
reserved3590MiB; GPU RNG capture contains one device. This is not a task score.
It ran the unchanged9fa9f57 model; DATA_PROVENANCE.json records matching formal
model/sampler hashes and the complete two-guard-only trainer difference. No
second smoke was run merely for this bookkeeping extension.

CPU checks:41 passed,2 skipped (optional pre-existing cases). Local environment
warns that optional torch-sparse cannot load; these CHM paths do not use it.
All eight BEST checkpoints and their exact selected predictions are retained.
