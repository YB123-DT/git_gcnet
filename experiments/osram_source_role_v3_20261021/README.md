# V3 Hierarchical Source–Role Interaction

INTERNAL TEST-ORACLE SCREEN

Only two approved fresh seed66 experiments: CMUMOSI150 epochs (3e-4 for100,
then3e-5 for50); CMUMOSEI60 epochs (constant1e-3). Adam, batch32, weight
decay1e-5, original MSE, cyclic0–.7 GCNet random masks and original splits.
No added loss, no dual views, no Nested, no JEPA and no old SourceRelation.

V3 retains all12 role/source slots. Same-source Base–Gap pair messages precede
same-role cross-source attention. Role-specific Local queries read three
sources independently per role. Four zero-initialized64→512 decoders correct
Base/Gap forward halves before the unchanged Flat. No global1600-d residual.
Only current-missing/real-history/padding-safe corrections are allowed.

Old source/experiments/checkpoints remain immutable. Source catalog, config,
data manifest, original initialization digest and exact fixed evaluation masks
are recorded. Training mask traces/update counts/LR schedules match Flat.

## Execute in an immutable source snapshot on biggpu

Use the original s0 Python environment and a verified healthy GPU UUID; host
GPU4 is forbidden. Root chooses/records the actual UUID, SHA and output path.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_source_role_v3_20261021.run \
  --source-commit SOURCE_SHA --gpu-uuid HEALTHY_GPU_UUID \
  --output /data2/yb/remote_experiments/osram_source_role_v3_20261021/execution_v1
```

The controller first runs four real optimizer updates across two short epochs
for MOSI (only two batch32 updates per epoch), and three updates for MOSEI. It checks
zero-init prediction parity, original/new parameter updates, upstream gradient
groups, finite values, one Memory scan per batch, full checkpoint restoration
and exact prediction replay. The two zero-initialized layers are never manually
opened. Gradient evidence is cumulative: full coverage is required after MOSI's
second epoch and MOSEI's first; formal training schedules are unchanged.
It then launches exactly two concurrent jobs,
subject to measured GPU/host-memory headroom. It never kills other processes,
changes batch size, expands seeds or tunes hyperparameters.

Every run preserves8 BEST weights with same-call predictions/labels/exact masks,
plus full last model/Adam/RNG/history. `worker --resume` requires exact identity.
Selected per-rate Test W-F1 means are not single-checkpoint deployment scores.
The controller produces completed-only RESULT.md/REPORT.json/METRICS.csv.

Protected Flat references are inherited from the reviewed SourceRelation worker;
new V3 code lives only in this independent experiment directory and new module.
