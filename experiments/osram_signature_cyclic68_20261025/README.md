# MOSI68 Signature: cyclic TRAIN, current-rate-only TEST

INTERNAL TEST-ORACLE SCREEN. Eight finite runs, no extra seeds or models.
Each run reuses the SAME original per-rate frozen Flat BEST and exact TEST masks
as the completed fixed-rate Signature68 control. Only stage2 training-rate
selection changes: original BalancedBatchRateSchedule over0.0–0.7 replaces
the single-rate selector. Each run still evaluates/selects ONLY its own rate.
This is not a newly trained Flat or one-stage joint model.

Fixed: cmerc-repro,biggpu,seed68,batch32,Adam WD1e-5,original MSE includinggold0,
150epochs,LR3e-4 for100 then3e-4/10 for50. Exactly300optimizer updates per run.
Unchanged SignatureBlock/window8/projections/zero exit to original pre-LNanchor.
Freeze encoder/CHM/Flat/LayerNorm/taskhead; task gradients pass through frozen
Norm/head to Signature. No CHM/write/query changes and no new loss.

GCNet-random-mask-v1 sampling remains unchanged within each selected rate.
Stage2 uses original cyclic rate scheduling, not uniform random rate draws.
All8parents see the same cyclic mask trajectory; each TEST only uses its
corresponding saved mask bank. Initial parent prediction replay, selected branch
recovery, frozen-parent digest/no-grad and actual train-rate coverage are checked.

Baseline comparison uses completed fixed Signature68 at
`/data2/yb/remote_experiments/osram_signature_stage2_extension_20261025/execution_v2/train/CMUMOSI_68_RATE`.
The original Flat and fixed Signature W-F1 means are79.074201 and79.044971.
Keep all positive/negative rates in the comparison. Same seed and unchanged
constructor source reproduce the initialization convention; the historical
fixed run did not store a separate initial-module digest, which is disclosed.

Run from an immutable snapshot using cmerc-repro:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_signature_cyclic68_20261025.scheduler \
 --root /data2/yb/remote_experiments/osram_signature_cyclic68_20261025/execution_v1 \
 --source-commit ACTUAL_SOURCE_SHA \
 --gpu2-uuid GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --gpu3-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

One real full-batch two-update smoke precedes8parallel150epoch jobs. Physical
GPU2/3,max4jobs each,UUIDbinding and live memory admission; GPU4 forbidden.
Do not stop,move or modify the ongoing MOSEI queue or anyone else's jobs.
Persistent QUEUE records allPIDs/logs/statuses; failures stop admission rather
than silently changing hyperparameters. Full branch/Adam/epoch/RNG last state
and immutable selected predictions/checkpoints support audited explicit resume.

Final RESULT.md/JSON/CSV compares original Flat,fixed-rate Signature andcyclic
Signature for all8rates plusmean/high. Per-rate Test-oracle and the additional
stage2 budget remain internal exploration, not independent test evidence.
