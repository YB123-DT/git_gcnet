# TASR one-stage joint training

INTERNAL TEST-ORACLE SCREEN. User authorized ONE MELD seed66 run,2026-10-19,
then explicitly changed selection to per-rate Test-best before any training.

This is NOT frozen-Adapter optimization and NOT unfreezing a trained Flat.
Initialize ObservedSetEncoder, original CHM, Local/Skip, Flat/head and TASR
from scratch; optimize all original trainable parameters and TASR together.
No pretrained weights enter the model. TASR keeps zero-initialized score output.
Original .6 joint Block Delta Write, queries, Base/Gap and Flat remain unchanged.
Source tracking denotes additive Value-slot contributions, not pure modalities.

## Fixed configuration

- Accepted MELD ECERC feature SHA256:
  `a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.
- Official train/dev/test; seed66;50epochs (original MELD joint-training budget).
- Adam LR1e-5/wd1e-5/batch32, original dropout/clip, constant LR, no new groups.
- Only original emotion CE; original cyclic random rates0.0–0.7,
  gcnet-random-mask-v1 host/guest padded-rectangle sampler unchanged.
- No Persistent training, second view, JEPA, completion, Gate, new loss or graph.
- Each rate independently selects its full-model BEST on test ACC; strict
  improvement only, preserving the earliest tie. UA/W-F1/CE and exact masks,
  logits, labels/IDs are saved from that SAME already executed test evaluation.
  Never resample a mask to fill UA next to an earlier best ACC.
- Original trainer evaluate_rate is called once per rate per epoch, consuming
  original GCNet sampling RNG normally; no frozen-mask or validation selection
  is silently substituted. The eight-rate mean is descriptive Test-oracle,
  not single-checkpoint deployment or independent generalization evidence.
- Explicit original current trainer's model-init RNG key is preserved:
  `missing_m3_model_init:fold:5` (also used for MELD in its actual code).
  No inference builder or frozen-Adapter setup helper is used to train.

## Implementation and recovery

`run.py` reuses the current trainer's actual `train_epoch`, preserving its
train mode/dropout, loss, clipping and optimizer step. A read-only scoped mask
observer records actual training mask hashes without sampling new masks.
Every batch must have exactly one forward and optimizer update.
Read-only gradient audits cover Encoder, Key/Value, Query, Flat, classifier,
TASR; initial zero gradients are recorded honestly, not treated as nonzero.

Eight best_miss_*.pt contain full model, epoch/metrics and exact artifact refs;
last.pt contains FULL model, Adam, epoch/history, RNG and all per-rate selection
state, effective config, source/data/split/mask identities. Explicit same-source
`--resume` restores everything; it does not load a pretrained Flat anchor.
BEST is saved before LAST. All source files are hash-checked at completion.

Protected historical checkpoints are inspected only to resolve and verify the
previous exact evaluation masks and labels/IDs; their weights are never loaded
into this model. Final PAIRED_MASK_REPLAY uses those masks, separately from the
exact Test-best result: random8 uses each rate's selected checkpoint; Persistent7
uses the new miss=.3 BEST for every pattern. This replay cannot overwrite or
be substituted for Test-best results, which come directly from selected records.
Branch-on/off share the SAME jointly trained checkpoint: `F0` in the reused
evaluation JSON means **TASR branch off**, NOT original separately trained Flat.
Do not compare it to frozen F0 as if the underlying parameters were identical.
The historical61.557 is a three-seed baseline, not this single-seed result.
Different runs' Test-best masks may differ; consult the separate paired replay.

## Execution

biggpu, `/data2/yb/conda_envs/cmerc-repro/bin/python`, healthy physicalGPU1 UUID
`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`. Never GPU4. One process only.
Use a new immutable source snapshot and independent output on the home disk.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_tasr_joint_20261019.run \
  --phase smoke --source-commit <actualSHA> \
  --output /home/qinjinghui/remote_experiments/osram_tasr_joint_20261019/smoke_v1 \
  --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
# After this finite5-batch real-data correctness check passes, initialize anew:
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_tasr_joint_20261019.run \
  --phase train --source-commit <sameSHA> \
  --output /home/qinjinghui/remote_experiments/osram_tasr_joint_20261019/runs_v1/MELD_seed66 \
  --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
```

CPU tests verify fresh model vs baseline common weights/RNG/logits, joint
updates, complete optimizer coverage/recovery, original trainer single-view
execution and smoke slicing. Real smoke must establish finite nonzero gradients
after several steps in all active groups, changed original backbone, exact
initial logits parity, real peak memory and runtime. It is not a performance run.
Formal training saves paired replay separately after50epochs.
No expansion to F1, other learning rates, seeds or datasets is authorized here.
