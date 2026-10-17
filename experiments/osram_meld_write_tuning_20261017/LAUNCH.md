# Five-job launch verified

RUNNING,NOT COMPLETED. INTERNAL TEST-ORACLE SCREEN.
Verified on biggpu at2026-10-17T12:09:45Z. Every job has completed a genuine
training epoch with nonzero task loss and jepa=0;none is merely a queued PID.

| Job | write_step | write_ridge | Host GPU | Python PID | Committed epoch at12:09:31Z |
|---|---:|---:|---:|---:|---:|
|W0|0.6|0.001|5|2861960|3/50|
|W1|0.3|0.001|0|2866617|1/50|
|W2|0.9|0.001|0|2866627|1/50|
|W3|0.6|0.0001|5|2866677|1/50|
|W4|0.6|0.01|0|2866859|1/50|

All five initial model hashes are exactly
c2ea3d6fdf2d6e559283a5bd15e4e2b3bea239b2e95aba161f5a271102402008,
with13,290,795parameters. Epoch1training mask/ID/speaker hashes agree exactly
across all five runs,covering9,989training utterances. All eight epoch1selection
mask hashes also agree across all five runs,each covering2,610test utterances.
These are actual-stream checks,not assumptions from equal seeds. The controller
will check complete-stream equality again after all five jobs finish.

Peak allocated memory observed after committed epochs was559.6–560.9MiB/job;
CUDA context/caching adds device memory not included in that allocator measure.
The scheduler retains at least4GiB/job reservation and refreshes resources before
each launch. Both physical cards are Tesla V100-SXM2-32GB. GPU4is excluded.
No unrelated task was killed or modified. Prelaunch free disk was23GiB;RAM
available was about225GiB. Temporary-version compaction remains confined to
these new runs while BEST/full-last checkpoints are retained.

## Provenance and persistent runner

- Driver commit:568daacb37a97f6d52dc52318e9e5537f470ad88.
- Original numerical/loader commit:b4a313841f098487a6482d2712bdac94aa8e319e.
- Environment:/data2/yb/conda_envs/cmerc-repro/bin/python;Python3.10.13,
  Torch1.13.1+cu117,NumPy1.26.4. No package/environment changes.
- Original ECERC feature SHA256:
  a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80.
- GPU5:GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62.
- GPU0:GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45.
- tmux:meld_write_tuning_v1;controller PID2861610.
- Source:/data2/yb/remote_experiments/osram_meld_write_tuning_20261017/driver_v1.
- Output:/data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1.
- Controller log:adjacent controller_v1.log.
- Per-job logs:runs_v1/queue/logs/W*.attempt1.log.

```bash
cd /data2/yb/remote_experiments/osram_meld_write_tuning_20261017/driver_v1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_write_tuning_20261017/run.py \
  --model-source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138 \
  --output /data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1
```

Do not rerun this command into an existing output or edit the frozen source.
LAUNCH_QUEUE.json and LAUNCH_RUNNER.json are startup snapshots,not live progress
or final outcomes. Actual states remain on the server at the paths above.

The automatic controller stops after exactly these five runs and writes
RESULTS.json,including failures and paired checks. It does not launch alpha,
Gap-strength,Optuna or multi-seed work. Runtime performance values are not
reported here as final results;no change to the accepted three-seed baseline
is made by this launch. Source remains Test-oracle exploration,not an
independent-test generalization claim.

Before deployment:28CPU tests passed,2pre-existing optional cases skipped;
Python compilation and git diff checks passed. Actual launch additionally
verified all five live workers,first-epoch updates,eight-rate evaluation,
selected checkpoint/artifact score parity,initialization equality and identical
first-epoch train/test mask streams. Complete50-epoch results are still pending.
