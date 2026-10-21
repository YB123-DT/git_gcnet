# MOSI Flat: two intermediate learning rates

INTERNAL TEST-ORACLE SCREEN

Only two authorized new jobs: original Flat, MOSI seed66,150epochs from
scratch. First100epochs use5e-4 or3e-4,final50 use5e-5 or3e-5 respectively.
No Router,new block,new loss,new seed or additional search.

Unchanged: cfg84 no-JEPA Large Flat1600,Local Skip,8heads×64 K/V,
causal joint Block Delta write0.6,batch32,Adam,WD1e-5,original dropout0.5/MSE,
cyclic0.0–0.7,original `gcnet-random-mask-v1`,same data/environment.
Per-rate earliest Test-best W-F1; ACC from the same selected predictions.
This eight-rate mean combines eight checkpoints,not one deployment model.

Reuse the completed original seed66 Flat as read-only parity reference:
`/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat`.
Verify initial backbone,split,evaluation masks,numerical core compatibility,and every
epoch's actual training-mask hash/optimizer step count. No old weights are
loaded for training initialization. Historical80.747 used another missing
sampler and is not the paired control here. The original reference predates the
opt-in source-role integration. Its two missing config fields are normalized
only to disabled defaults;exact before/after SHA256 pairs explicitly allow
the three reviewed integration files. Unknown source changes are rejected.
Do not label these source files byte-identical. Disabled forward/backward/RNG
equivalence is checked against the original27b8c1f implementation;the actual
protected checkpoint prediction bridge remains mandatory before training.

Only the existing experiment worker's LR setter gains an explicit final LR;
its default remains constant1e-5 (and the existing2e-5 wrapper stays constant).
No model,mask sampler,optimizer implementation or task loss is edited.
Existing same-Flat GPU validation and peak1245.2MiB are reused; no redundant
standalone GPU smoke. CPU tests cover schedule boundaries and legacy defaults.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_mosi_mid_lr_20261021.run \
  --output /data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

Run from an immutable source snapshot on biggpu under tmux. GPU7 selected
after live resource inspection;physicalGPU4 forbidden. At most two concurrent
jobs,reserve4GiB/job plus2GiB card margin,64GiB available RAM,12GiB free disk.
Do not kill peers or change batch size. Queue and child locks prevent duplicate
submission;explicit `--resume` requires matching full model/optimizer/RNG state.
Failures are recorded and never silently retried. Existing results are read-only.

Each run retains eight BEST checkpoints,matching predictions/masks,full last,
config,history,source/data/environment hashes and GPU UUID. On both runs'
completion,the controller writes RESULT.md,SUMMARY.json,METRICS.csv after
recomputing selected metrics and verifying same-cohort/mask comparisons.
Old and new scores remain explicitly exploratory Test-oracle,not paper claims.
