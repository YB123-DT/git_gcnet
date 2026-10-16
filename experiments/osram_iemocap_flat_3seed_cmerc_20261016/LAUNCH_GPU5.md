# Original Flat completion on GPU5: observed launch

INTERNAL PER-RATE TEST-ORACLE ACC. Training in progress; no new final scores.

The user's latest instruction resumes original IEMOCAP4/6 Flat baseline
completion on **GPU5, two concurrent fold jobs**. It supersedes the earlier
pause and never authorizes a B6/MELD rerun. Reuse existing Flat66; complete
Flat67/68, full fivefold,100epochs each:20newfoldjobs, not20newmodels perrate.
Every fold trains one cyclic0.0–0.7 model and retains8rate-selected BESTs.

- Implementation SHA: `9a70b1f2f57773e5a4c034d31dc923695ab8b0e6`.
- Numerical model/trainer SHA: `785a01242ef1a82b97d5337acbb33ba963df5537`, unchanged.
- Server: `ssh biggpu`.
- Immutable implementation: `/home/qinjinghui/remote_experiments/osram_iemocap_flat3_gpu5_code_9a70b1f`.
- Output: `/home/qinjinghui/remote_experiments/osram_iemocap_flat_3seed_cmerc_20261016`.
- tmux: `iemocap_flat3_gpu5_1016`; controller PID: `3834416`.
- GPU5 UUID: `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`.
- Verification:31focused CPU tests passed, compilation/diff checks passed.
  One pre-existing optional torch-sparse warning does not affect this CHM path.

Actual initial queue:2running,18pending,0failed. Independent outputs:

| Dataset | Seed | Fold | Physical GPU | Worker PID |
|---|---:|---:|---:|---:|
| IEMOCAPFour |67|5|5|3835143|
| IEMOCAPSix |67|5|5|3835181|

Both worker summaries confirm `osram_meaningful_block=none`,
`gcnet-random-mask-v1`, original Local Skip and the accepted runtime:
`/data2/yb/conda_envs/cmerc-repro/bin/python`, Python3.10.13,
Torch1.13.1+cu117/CUDA11.7, NumPy1.26.4, sklearn1.3.2. The launcher and
workers reject incompatible runtimes. No experiment was added to another
GPU; unrelated processes were not signalled. Admission may lower concurrency
if memory is insufficient, but never changes batch32 or training parameters.

The existing longest-batch B6 smoke bounds memory conservatively; no extra
standalone GPU smoke or tuning was run. These code edits only change resource
admission, seed permission and evidence recording, not OSRAM/Flat mathematics.

Command, from the immutable implementation checkout, run persistently in tmux:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_iemocap_flat_3seed_cmerc_20261016.run \
  --worker /home/qinjinghui/remote_experiments/osram_iemocap_flat3_gpu5_code_9a70b1f/experiments/osram_iemocap_b6_transfer_20261015/worker.py \
  --smoke-summary /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015_preflight/smoke_v2/SUMMARY.json \
  --data-provenance /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015_preflight/DATA_PROVENANCE.json
```

`MANIFEST.json` pins all20newconfigs and hashes the10reused66summaries, source,
worker and data provenance. `RUNNER.json`, `queue/QUEUE.json`, controller and
worker logs are live authorities. All BESTs, matching selected predictions,
availability and complete last-training states must remain retained. Every
new fold must pass its actual first-epoch artifact gate before epoch2.

Observed after launch: both initial workers reached epoch4/100 and passed
`FIRST_EPOCH_RETENTION_CHECK.json`:8BESTs,8matching selected artifacts and
1CUDA RNG state each. There are16canonical BEST files and2last-training
states in the new output. Queue remains2running/18pending/0failed. These
early epochs confirm execution/retention only, not eventual model performance.

Fivefold results require all5folds; final three-seed statistics require all30
validfoldresults including the10read-only reused66folds. Do not report partial
fold averages, claim the new scores are already known, or replace old results.
