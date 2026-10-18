# Resource-only expansion to healthy GPU0/2

User requested other cards rather than waiting for occupiedGPU2. Fresh check:
GPU0 andGPU1 idle32GB; GPU4 forbidden. B1/B2 already completed onGPU0.
Selected GPU0 for the3 waiting WD runs; existing WD0 staysGPU2 unchanged.

Resource wrapper commit829465d4f955f68575587fb36fff478f56511f47,
standalone deployment:
`/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/resource_source_v1/resources.py`.
It imports original sealed source3628d1e, verifies its original SHA256 and
SOURCE_MANIFEST, and changes ONLY runtime GPU whitelist/scheduler selection.
Original model/trainer/config/optimizer/RNG and checkpoint identity are not
edited or rewritten. Wrapper SHA and commit are saved separately in
RESOURCE_CONTROLLER.json and each newly bound run's RESOURCE_OVERRIDE.json.

Queue handoff stopped ONLY verified owned controller313773 (SIGTERM, not
process group); child workers use separate sessions and remain untouched.
Its command's resolved source path, ownerUID and non-worker role were checked
before signalling. Original WD0 PID313904 remained alive and advanced epochs.
No other user's controller or training process was signalled.

Same persistent queue/output is now managed through tmux `meld_wd66_gpus_1018`.
Original running child recognized by birth/PID/exactargv, not relaunched.
Only waiting jobs receive new wrapper commands and resume their full epoch4
checkpoints. Original guarded50epoch training, selection and recovery paths
remain active; no new smoke, optimizer or model change.

| WD | physicalGPU | PID |
|---|---:|---:|
|0|2|313904 (unchanged)|
|1e-5|0|516085|
|1e-4|0|516089|
|5e-4|0|516093|

PhysicalGPU0 UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45;
GPU2 UUID GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4; each child logicalcuda:0.
All4 observed running together. GPU4 never used. No checkpoint deleted.
64 resource/config/old-worker regression tests passed (TDD missing adapter
failures first). New source scheduling wrapper did not replace the frozen
numerical source. GPU placement is recorded, not claimed bitwise equivalent
across hardware; actual50epoch mask streams are still checked against O6.

Launch from original source directory:

```
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
 /home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/resource_source_v1/resources.py \
 --output /home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1 \
 --wrapper-commit 829465d4f955f68575587fb36fff478f56511f47
```

All new workers are same4 previously approved runs, not additional experiments.
