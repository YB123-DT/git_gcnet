# Frozen original Nested: MOSEI / IEMOCAP-6 / MELD

Finite user-approved continuation of the completed MOSI66 frozen Nested test.
Only seed66, no new baseline training or HPO. Old code/results/weights untouched.

| Dataset | Folds | Rates | Formal jobs | Stage2 epochs | Stage2 LR | Selection |
|---|---:|---:|---:|---:|---:|---|
| MOSEI |1|8|8|60|3e-4 constant|Test-best W-F1|
| MELD |1|8|8|50|3e-4 constant|Test-best ACC|
| IEMOCAP-6 |5|8|40|100|3e-4 constant|Test-best ACC|

The original Flat parent LR remains MOSEI1e-4/MELD1e-5/IEMOCAP1e-3;
it is not the new branch LR. All parents and their exact saved GCNet masks are
the protected accepted cmerc-repro versions used by completed cyclic Signature.
Class UA/W-F1 use the ACC-selected same epoch and prediction, never a new random
evaluation or a separately UA-selected checkpoint.

## Model and training

Reuse `osram_frozen_nested_gcnet_20261025.model.FrozenNestedModel` unchanged.
Only existing159235parameter `nested_gnn_rooted_evidence` trains, dim64,
depth3,MLP64,8actualmemoryheads. It corrects Local/Base/Gap at the original
Flat input; raw LocalSkip remains unchanged. Freeze/eval original Encoder,
CHM,queries,Flat1600,LayerNorm and taskhead. One causal live CHM scan, joint
ATVwrite,read-before-write,write_step0.6 unchanged. No Signature features,
root-aware topology,Gate,completion,JEPA or auxiliary loss.

Adam,batch32,WD1e-5; original cyclicGCNet-random-mask-v1 training. Every own-rate
run evaluates only its protected current-rate saved mask. MOSEI original MSE
includes neutral labels; IEMOCAP/MELD original CE. Extra training follows each
dataset's completed Signature budget, enabling a matched stage2 comparison.

**INTERNAL TEST-ORACLE SCREEN.** Each rate has its own frozen parent/block pair;
eight-rate means are not one deployed model. IEMOCAP headline is equal fivefold
mean; separately labelled pooled out-of-fold ACC/UA is also saved. Only complete
fold/rate groups are aggregated. Seed66 does not establish independent gains.

## Immutable numerical sources and preparation

MOSEI/MELD run cwd/source:
`/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
IEMOCAP runs cwd/source:
`/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/source_21e46f2`.
The new committed overlay contains only this experiment and the reviewed
MOSI frozen wrapper / shared stage2 utilities. Workers must run from their own
dataset's pinned numerical cwd, not an editable live checkout.

Bind each dataset environment before global trainer/config imports. Validate
parent dataset/config/split/environment/source/featureSHA, strict checkpoint
load, exact original selected logits/classes and mask replay. Formal sampling,
rate,utterance and update traces must match the existing same-rate Signature
history; independent test-mask draws are prohibited. Initial new predictions
must exactly equal Flat. Strict reload and all saved arrays must replay.

## Finite persistent launch

Run cmerc-repro with immutable overlay first in PYTHONPATH. Add the pinned
IEMOCAP source as a fallback for the master summarizer's original metric helpers;
workers set their own cwd so their actual models resolve from the correct source.

```sh
python -m experiments.osram_frozen_nested_cross_20261025.scheduler \
  --root /data2/yb/remote_experiments/osram_frozen_nested_cross_20261025/execution_v1 \
  --source-commit ACTUAL_SOURCE_SHA \
  --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
  --per-gpu 4
```

One real two-update GPU smoke per dataset gates formal admission. PhysicalGPU0/
1/2 only,max4percard,max12total,staggered starts and measured GPU/RAM reserves.
The two-update smoke does not establish worst-case missing-heavy backward peaks.
Initially admit one formal job per dataset; after its actual history covers all
eight training rates, expand concurrency using observed formal resource peaks.
GPU4 and unrelated processes are untouched. Insufficient resources queue without
changing batch or architecture. tmux holds the controller; QUEUE.json preserves
commands/PIDs/cwd/UUID/logs/status. Exact compatible block/Adam/RNG/history
recovery uses `--resume`; failed artifacts are never silently overwritten.

Automatic final RESULT.md/JSON/CSV independently verify selected checkpoint/
prediction hashes,protected parent arrays and metrics,full epoch budget,
earliest strict selection and frozen gradient/state checks. Per-rate class
recall/support,corrections/harms,cost and all negative/failed cases remain visible.
No model weights,feature caches or large arrays are committed to GitHub.
