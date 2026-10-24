# MOSEI Flat write-strength screen

INTERNAL TEST-ORACLE SCREEN. Four new independent seed66 runs only.

| Write step | Action |
|---:|---|
| .2 | fresh training |
| .4 | fresh training |
| .6 | reuse completed paired original Flat |
| .8 | fresh training |
| 1.0 | fresh training |

Original Flat1600, cmerc-repro Torch1.13.1+cu117 / NumPy1.26.4,
constant LR1e-4,60epochs,batch32,Adam,WD1e-5,MSE,global clipping1.0.
Original gcnet-random-mask-v1 cyclic0–.7, original splits/features and exact
test-mask bank; per-rate TEST-best W-F1 with same-call ACC/MSE and earliest tie.
No Nested, Gate, V3, JEPA, auxiliary loss, new missing augmentation or second view.
CHM remains8x64, joint observed A/T/V Block Delta Write, read-before-write,
unchanged Key/Value/Query/Base/Gap/LocalSkip/Flat/task head. Only the original
`osram_write_step` scalar changes; no mathematical write formula changes.

The default-disabled setup callbacks in the accepted cmerc worker are reused.
All initial parameters must byte-match the untrained original factory AND
the completed paired Flat initialization digest. This is fresh training,
not loading a trained0.6 checkpoint or resuming s0 weights. Each real OSRAM
instance's `write_step` is checked before the first training batch.

Read-only0.6 control:
`/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none`.
Control mean ACC83.626857/W-F1 83.477635/high W-F1 79.898427.
It has the same UUID CUDA RNG repair, unlike older unrepaired cmerc results.

Server biggpu, healthy GPU7 only, UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`; process device cuda:0.
At most4 own workers,6GiB full-phase reservation each plus2GiB margin;
>=64GiB available system RAM and>=25GiB disk. Unknown foreign allocations
remain charged. Resource limits can reduce parallelism; no batch/precision/
model change, foreign process signals, or GPU4 use. Existing GPU0/1 capacity
experiments are not moved, restarted, or edited.

Immutable source is a copy of the corrected cmerc capacity snapshot plus only
this standalone experiment/tests. No running source is edited. Every Python
and AGENTS file is hashed before admission and checked on explicit recovery.
All eight BESTs, matching predictions/masks, complete model/Adam/CPU+CUDA RNG
last checkpoint and per-epoch history remain remote. No automatic failed-job
retry, seed expansion, continuation beyond60epochs or HPO.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_mosei_write_step_20261024.run \
  --output /data2/yb/remote_experiments/osram_mosei_write_step_20261024/execution_v1 \
  --source-commit RUNNER_SHA
```

The tmux controller saves QUEUE.json, independent worker logs and failure
states, then automatically validates all completed60epoch histories, matching
training masks/update counts/LR, environment/data/split/test-mask/source and
initialization parity against0.6, selected artifact hashes, earliest W-F1
selection and aggregate arithmetic before writing RESULT.md/JSON/PER_RATE.csv.
Incomplete or failed arms are never ranked. Test-best means may combine eight
different checkpoints: exploratory results only, not independent generalization
or single-checkpoint deployment performance.
