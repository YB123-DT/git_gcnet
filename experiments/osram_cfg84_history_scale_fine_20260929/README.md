# Frozen Flat fine history-retention scan

User-requested continuation of the coarse history-scale diagnostic.
No weight fine-tuning or retraining. Original Flat checkpoints, seeds
66/67/68 and eight random-missing rates are unchanged.

New alpha values: .8, .85, .9, .95, 1.05, 1.1, 1.15, 1.2.
Alpha=1 is additionally reevaluated as a reproduction/invariance control.
Total 216 evaluations, including 24 control evaluations.

Run on biggpu host GPU0 (UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45),
PID 3470781; initial GPU free memory 31755 MiB, utilization 0%.
Remote root `/data1/yb/remote_experiments/osram_cfg84_history_scale_fine_20260929`.
Isolated `code/` snapshot copied from the completed coarse scan, with only
the diagnostic runner updated to accept `--fine`. Original coarse code and
results are untouched. Log: `run.log`; outputs: `results/`.

From its isolated code root:

```bash
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_cfg84_history_scale_20260929/run.py --fine
```

All alphas share identical checkpoint, ordered labels, masks and pre-scaling
Local/Base/masked-Gap tensor hash within each seed/rate. The script asserts
these invariants and control W-F1 reproduction. It refuses to overwrite
existing outputs. No memory, query, key/value, task-head or mask changes.
Alpha is NOT a forgetting/write parameter. Alpha>1 amplifies the read values.

Internal Test-oracle diagnostics only. The best test-alpha is not a validated
hyperparameter and must not be reported as an official improvement. All nine
settings will be retained and reported. Across rates, historical checkpoint
epochs may differ; within each rate all nine alpha settings use one checkpoint.
