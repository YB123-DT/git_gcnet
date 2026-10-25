# Signature fixed-rate stage2: MOSI three seeds and MOSEI66

INTERNAL TEST-ORACLE SCREEN. This finite extension reuses completed MOSI66
Signature (ACC79.820884/W-F1 79.853859) and adds only MOSI67/68 and MOSEI66.
Each missing rate0–0.7 independently freezes its own original Flat BEST, trains
one unchanged SignatureBlock, and evaluates ONLY that same rate. No cyclic
stage2 masks, no NeuMiss/CSDI, no new Flat training or automatic search.

## Fixed settings and parents

| Dataset | Stage2 seeds | Branch epochs | Branch LR | Parent source |
|---|---|---:|---|---|
| MOSI | reuse66; add67,68 |150|3e-4 first100,3e-4/10 final50|accepted cmerc original Flat per-seed BEST|
| MOSEI |66 only|60|3e-4 constant|latest repaired-RNG cmerc original LargeFlat1600, parent LR1e-4|

All branches: batch32,Adam,WD1e-5,original MSE including gold0, original
GCNet-random-mask-v1 host/guest/padding behavior. Task-head polarity ACC/W-F1
exclude gold0 by the original evaluator. Signature retains strict-past window8,
per-modality256→8,degree2,time coordinate,zero basepoint,and zero exit to1600.
No module parameters or loss definitions change from completed MOSI66 stage2.
Only branch seed, parent identity, dataset and declared epoch budget vary.

MOSI67/68 parents:
`/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSI/seed_{67,68}/flat`.
Reuse66 stage2:
`/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1/train/signature_RATE`.

MOSEI66 parent:
`/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none`.
Its frozen parent mean W-F1 is83.477635; it is NOT the earlier unrepaired-RNG
unified cmerc parent83.288796. Exact common mask records remain in
`osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat/fixed_masks.pt`.
Both mask ledgers must agree, while predictions/weights must come from the
selected repaired parent. Never fill a paired table with the earlier score.

Each parent checkpoint SHA/epoch/rate/identity and current-source strict state
loading are checked. Source differences are recorded; only reviewed disabled
extensions are allowed. Saved original predictions/classes/labels/speakers/
utterance order/masks must replay before any branch training. The older source
hash gate is not silently bypassed without this explicit prediction bridge.

## Execution

Use immutable committed source and cmerc-repro Python on biggpu:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_signature_stage2_extension_20261025.scheduler \
 --root /data2/yb/remote_experiments/osram_signature_stage2_extension_20261025/execution_v2 \
 --source-commit ACTUAL_SOURCE_SHA \
 --gpu0-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
 --gpu1-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
 --gpu3-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

After unit checks, two separate real full-batch two-update smokes (MOSI67.7,
MOSEI66.7) gate formal admission. The controller launches only24 new jobs,
max4 per physical GPU0/1/3 and12 total. Start the8longer MOSEI jobs first,
then fill remaining slots with MOSI. Check actual free memory and smoke peaks,
never use GPU4 or modify other processes. Run persistently via tmux.

Each run saves fixed config,immutable source/parent/data/mask provenance,
per-epoch metrics,branch gradients,exact selected predictions,branch BEST and
complete branch/Adam/epoch/RNG last state. The original parents stay frozen
in eval mode and are referenced by hash, not overwritten. Resume only the same
unfinished identity. Selected task predictions must reproduce after strict
branch recovery and the parent's hash/grad state must remain unchanged.

Final result tables include all32 conditions (MOSI3×8 plus MOSEI1×8),per-seed
eight-rate and high-missing means,paired parent deltas,and MOSI sample SD across
three seeds. Rates are NOT independent seed repeats. MOSEI has no seed SD.
Selection is per-rate Test-best W-F1,earliest ties,so the mean is not one
deployment checkpoint. Additional training budget and compounded Test-oracle
selection must be disclosed; results cannot prove independent generalization.

## Retained preflight attempt

Execution_v1 (source e47c60f) stopped before formal training: MOSI67 smoke passed,
but MOSEI's source guard detected `meaningful_new40_registry.py`. An exact review
against immutable parent git3777ab1 showed only27 new literal Nested-sweep
entries, with the remaining AST identical. Flat's `meaningful_block=none`
does not instantiate this adapter. The corrected preparation accepts ONLY this
specific old/current SHA pair when the block is none; unknown changes still
fail, and full saved-prediction replay remains mandatory. Execution_v1 is
retained as failed preflight evidence, never included in performance results.
