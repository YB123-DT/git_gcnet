# Frozen original Nested: complete seeds66/67/68

User-approved replication only. Reuse the64 completed seed66 own-rate runs;
train128 new own-rate branches for seeds67/68. No parent retraining, architecture
changes, HPO, auxiliary losses or missing-protocol changes. Original Nested has
159235 trainable parameters; everything in the accepted original Flat is frozen
and eval, including Encoder, CHM, Flat1600, raw LocalSkip, Norm and task head.

| Dataset | New seeds | Folds | New branches | Branch epochs | Branch LR | Selection |
|---|---|---:|---:|---:|---|---|
| MOSI |67/68|1|16|150|3e-4 first100, then exact3e-4/10|Test-best W-F1|
| MOSEI |67/68|1|16|60|3e-4|Test-best W-F1|
| MELD |67/68|1|16|50|3e-4|Test-best ACC|
| IEMOCAP-6 |67/68|5|80|100|3e-4|Test-best ACC|

Adam,batch32,WD1e-5; original MSE including neutral regression labels / class CE.
Cyclic GCNet-random-mask-v1 training, independent host/guest original sampler,
padding unchanged. Each branch evaluates only its own rate using protected saved
Flat test masks. Class UA/W-F1 come from the same ACC-selected epoch/prediction.
Parent LRs remain MOSI3e-4,MOSEI1e-4,MELD1e-5,IEMOCAP1e-3, not the branch LR.

**INTERNAL TEST-ORACLE SCREEN.** Eight own-rate parent/block pairs are not one
deployable checkpoint. Equal-rate / equal-fivefold / three-seed mean and sample
std are explicitly separated; incomplete groups never produce a headline mean.
The extra stage2 budget and exploratory selection remain limitations.

## Protected inputs and immutable sources

Own-seed parent checkpoints, source configs, feature/environment hashes and
selected masks/predictions are verified before training. No new random test-mask
draw. MOSEI67/68 preserve their accepted common seed66 maskbank while using their
own weights. MELD67/68 use their own S67_WD1/S68_WD1 protected baseline entries.
IEMOCAP-6 includes all5 folds, not onlyfold5.

Numerical roots are unchanged historical snapshots:

- MOSI67/MOSEI67/68/MELD67/68: signature_cyclic_multi/source_e0ca94c.
- MOSI68: signature_cyclic68/source_4d5aa09.
- IEMOCAP-6: signature_iemocap_cyclic/source_21e46f2.

Nested implementation hashes agree across these snapshots. A new immutable
experiment overlay routes to the correct root per worker. Source/config/parent/
data/environment/mask/seed/fold/GPU identities are persisted per run. Completed
same-seed Signature traces are checked where they exist (112 branches). MELD67/
68 have no such completed reference; record that absence explicitly, validate
the original sampler and run an RNG-restored two-update sampler control in smoke.
Never substitute a seed66 trace as a seed67/68 trace.

## Execution

Environment `/data2/yb/conda_envs/cmerc-repro/bin/python`, server `ssh biggpu`.
Physical GPU0/1/2 UUID binding only; physical4 is forbidden. Max4 task processes
per card and12 total, further limited by measured GPU/RAM/disk capacity. Eight
real GPU smokes (dataset ×newseed, IEMO fold5) precede admission. A full actual
cyclic training cycle gates expansion; existing jobs and other users untouched.

The persistent tmux controller automatically fills free slots and summarizes.
Exact full resume restores block/Adam/RNG/history, checks immutable identity, and
never silently overwrites a completed run. A failure stops new admissions but
does not kill other live jobs.

```sh
python -m experiments.osram_frozen_nested_3seed_20261026.prepare --preflight
python -m experiments.osram_frozen_nested_3seed_20261026.scheduler \
  --root /data2/yb/remote_experiments/osram_frozen_nested_3seed_20261026/execution_v1 \
  --source-commit ACTUAL_OVERLAY_COMMIT \
  --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
  --per-gpu 4
```

Workers run from their own numerical root. Overlay PYTHONPATH must contain this
experiment, reused frozen Nested wrapper and shared stage2 package, then the
IEMOCAP numeric root as a master metric-helper fallback. Do not add current live
gcnet source to PYTHONPATH. QUEUE.json preserves PID,command,cwd,UUID,log,status;
SUMMARY/config/history/predictions/checkpoints remain remote. Final RESULT.md/
JSON/CSV recompute selected metrics, verify hashes and same-prediction alignment,
and combine64 protected66 +128 new runs. Only code/tests/small text statistics
are committed; no datasets,feature caches or model weights.
