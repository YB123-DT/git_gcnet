# GPU0 OOM recovery — incomplete experiment

The initial five-worker admission was insufficiently conservative: it used
PyTorch allocated-memory peaks without enough CUDA context/reservation margin.
seed68/fold5 exhausted GPU0 at the start of epoch12 (last completed epoch11).
The controller stopped admitting jobs and drained the four surviving workers;
all four completed100epochs. GPU0 was idle when the failure was investigated.
No unrelated job was killed. This is a resource failure, not a model score.

Completed eight-rate means at the investigation:

| Seed | Fold | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
|66|1|60.752|61.196|60.038|
|66|2|63.864|62.807|63.670|
|66|5 (reused)|65.689|63.895|65.748|
|67|1|61.499|61.332|61.162|
|67|5|65.357|62.434|65.013|

These are individual folds, NOT complete fivefold or three-seed averages.
Different test sessions must not be directly interpreted as seed/model gains.

`recover_oom.py` is an independent controller-only recovery entry. It imports
the ORIGINAL immutable `source_v1` code; no numerical/model/sampler file is
changed. It keeps the original queue/run identities and masks, restores only
the diagnosed68/fold5 failure from its complete `last.pt`, and requeues the nine
never-started jobs. Completed jobs and reused66/fold5 are never retrained.

Recovery admission caps at THREE workers, reserves8192MiB per worker plus2048MiB
global margin, and still checks external allocations/RAM/disk. No batch-size,
precision,LR,epoch,initialization or training-protocol adjustment. The historical
queue identity retains its requested six ceiling; the explicit `oom_recovery`
record overrides live admission to three. One audited retry only; another
failure stops admission rather than silently retrying.

Before admission, verify complete epoch11 history, selected artifacts, Adam,
one CUDA RNG stream and matching identity (JSON tuple/list representation is
normalized ONLY for comparing SUMMARY; saved state is not changed). Original
worker subsequently performs its strict identity check and full restoration.
Original failed QUEUE and SUMMARY are archived exclusively before requeueing.
27 focused tests passed, including all existing expansion tests and four
recovery-policy tests. Actual resumed progress must be verified separately;
tests alone do not establish successful recovery or final performance.

Actual recovery verified on2026-10-24: controllerPID2789485 in persistent tmux
`nested128_recovery_20261024`; seed68/fold5 PID2789927 completed epoch12 after
restoring epoch11. seed68/fold1 PID2790806 and seed66/fold3 PID2790990 each
completed epoch1. Queue is running, error null. Original failed QUEUE/SUMMARY
are retained as `OOM_FAILURE_QUEUE.json` / `OOM_FAILED_SUMMARY.json`.
Recovery script SHA256:
`1dca36027ea274627a248a71ff345997278b3428496ee15ddb6fe41b3f43acf6`.
Restored last.pt SHA256:
`34431a83e5eaa666a723dc70b783414732c3d51729ea7fc7679fc5defa8a6a99`.
No complete three-seed/fivefold mean exists yet. Original launch STATUS.md
and LAUNCH.json are historical receipts, superseded by this recovery record.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u /absolute/recover_oom.py \
  --source-root /data2/yb/remote_experiments/osram_iemocap6_nested128_3seed_20261024/source_v1 \
  --output /data2/yb/remote_experiments/osram_iemocap6_nested128_3seed_20261024/execution_v1
```
