# Actual fixed-LR weight-decay launch

INTERNAL TEST-ORACLE SCREEN; eight finite jobs, no added block.

- Implementation commit: `b222968a10b037902c5ae5c8441718f1dd054663`, pushed to
  `YB123-DT/git_gcnet`, branch `feature/ecerc-features`.
- Server: biggpu; original cmerc-repro environment and b4a3138 numerical source.
- Source snapshot:
  `/home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/source_b222968`.
- Output:
  `/home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/runs_v1`.
- Persistent tmux: `meld_fixed_lr_wd_1018`.
- Controller log: experiment root `controller_v1.log`.
- Worker logs: `runs_v1/queue/logs/{job}.attempt1.log`.
- Both shared and extension manifests passed. 57 new/legacy tests passed.

All8 jobs were observed running simultaneously, four on each healthy GPU.
Actual nvidia-smi total memory was about5.2GiB per card; both cards active.
GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`;
GPU2 UUID `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`.
GPU4 forbidden; existing GPU1/3/6/7 tasks were not modified or killed.

| Job | Seed | LR | WD | Physical GPU | PID |
|---|---:|---:|---:|---:|---:|
|S67_WD0|67|1e-5|0|0|2955611|
|S67_WD1|67|1e-5|1e-5|2|2955745|
|S67_WD2|67|1e-5|1e-4|0|2955833|
|S67_WD3|67|1e-5|5e-4|2|2955886|
|S68_WD0|68|1e-5|0|0|2955978|
|S68_WD1|68|1e-5|1e-5|2|2955987|
|S68_WD2|68|1e-5|1e-4|0|2955991|
|S68_WD3|68|1e-5|5e-4|2|2955995|

Same-seed WD2e-4 controls are reused from the completed small-LR three-seed task,
not retrained. Epoch observers check exact initialization, actual optimizer
LR/WD and training/test mask streams against the corresponding seed's control.
Per-rate BEST checkpoint and exact winning-epoch mask/logits/labels are saved,
with UA/W-F1 from those same predictions, plus complete recovery state.

This is a launch record, not a final performance report. The queue automatically
writes all8 results/failures and per-WD two-seed means, then stops. It does not
train seed66 or label two-seed means as three-seed results.

All8 jobs subsequently committed epoch1 after real training/backward/optimizer
steps and eight-rate evaluation. Each passed the initialization hash and actual
train/test mask-stream checks against its same-seed control. All three optimizer
groups used LR1e-5 and the declared WD, including exact0 for WD0. Each output
had8 BEST checkpoint files and complete last_training.pt. No failure was
observed. The timestamped early snapshot is in LAUNCH_STATUS.json; it is not
a live progress file or a claim of completed50epochs.

## Reproduce

From the immutable source snapshot (do not duplicate an existing active run):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_fixed_lr_wd_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/runs_v1 \
  --driver-commit b222968a10b037902c5ae5c8441718f1dd054663
```
