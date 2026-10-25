# Frozen Signature: cyclic stage2 extension

INTERNAL TEST-ORACLE SCREEN. This is the unchanged Deep Signature branch,
not a new Flat training, not joint end-to-end training, and not NeuMiss/CSDI.
The completed MOSI68 cyclic experiment remains untouched. This finite queue adds:

| Dataset | Seeds | Independent rate-specific branches | Epochs per branch | Branch LR |
| --- | --- | ---: | ---: | --- |
| MOSI | 66,67 | 16 | 150 | 3e-4 for100,3e-5 for50 |
| MOSEI | 66,67,68 | 24 | 60 | 3e-4 constant |
| MELD | 66 | 8 | 50 | 3e-4 constant |

Each run freezes its own existing per-rate original Flat BEST and trains only
Signature on the original cyclic batch-rate schedule over0.0–0.7. It evaluates
ONLY its designated rate, using that parent's exact saved evaluation mask.
Eight branches are therefore eight separate models, not one deployment model.
Parent/dropout/evaluation remain frozen; original CHM, Local/Base/Gap, joint
write, Query, Local Skip, Flat1600 and task head are not changed. Original
GCNet-random-mask-v1 host/guest/padding semantics and original splits are kept.

All new branches use batch32,Adam,WD1e-5,seed matching their parent. Parent
training LR is not the branch LR: MOSEI frozen Flat used1e-4; MELD frozen Flat
used1e-5/WD1e-5. The predeclared branch LR3e-4 matches the preceding stage2
Signature experiments; this queue does not search LR or alter the parents.
Sentiment uses original MSE (including gold0) and per-rate Test-best W-F1;
polarity ACC/W-F1 omit gold0. MELD uses original seven-class CE and Test-best
ACC, with UA/W-F1/CE from that SAME epoch, prediction and saved mask. Ties
keep the earliest epoch. No new test-mask draws occur during evaluation.

The strict-past window8 degree2 Signature implementation is reused without
changes. It has281856 parameters and a zero output projection to the original
1600-dimensional pre-LayerNorm anchor. No-history/padding corrections are zero.
The first prediction must equal the selected frozen parent. Before training,
strict checkpoint loading and saved parent prediction/mask replay are required.
Selected branch checkpoint replay, frozen-parent digest/gradient checks, finite
loss/gradients and actual cyclic mask/rate traces are recorded.

New code is isolated here. The preparation layer pins the accepted sentiment
parents and protected MELD baseline; no existing result or checkpoint is
overwritten. Unknown source/config/data differences fail preflight. Explicit
reviewed inactive-source differences are recorded alongside the numerical
saved-prediction replay proof.

## Persistent execution

Run from a committed immutable snapshot on biggpu with cmerc-repro:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_signature_cyclic_multi_20261025.scheduler \
 --root /data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/execution_v1 \
 --source-commit ACTUAL_SOURCE_SHA \
 --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4,3=GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a \
 --per-gpu 6
```

One real two-update smoke per dataset gates the finite48-job queue. A live
resource check precedes staggered admission. Six perGPU is a ceiling, not a
promise: measured VRAM/RAM and reserve can reduce concurrency without changing
batch or model dimensions. Physical GPU4 is forbidden, GPU5/6 and other tasks
are untouched. Controller lives in tmux; QUEUE.json retains commands/PIDs/logs/
GPU UUIDs/status. Explicit `--resume` only recovers compatible full last.pt
state (branch,Adam,epoch,RNG); live children are not duplicated.

Per-run outputs include immutable source and parent hashes, effective config,
exact mask hash,initialization digest,all epoch metrics,selected predictions,
BEST and complete last state. Final RESULT.md/JSON/CSV report each rate,each
seed,eight-rate/high means and own-parent paired changes. Incomplete jobs
are not ranked or silently included. These exploratory Test-oracle scores
and additional training budgets are not independent generalization evidence.
