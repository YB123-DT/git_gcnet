# MOSI: restore per-rate Test-best, retain original GCNet missing

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

User correction on2026-10-20: restore the former checkpoint selection policy.
This finite rerun covers only MOSI seeds66/67/68 × Flat/one-stage Router,
six fresh150epoch trajectories. MELD/IEMOCAP are not silently retrained here.
One trajectory per seed/model, NOT eight independently trained missing-rate models.

## What changes and what does not

- Change: each missing rate independently saves its best TEST W-F1 epoch.
  Ties keep the earliest epoch, without MSE tie-breaking.
- Keep: `gcnet-random-mask-v1`, original host/guest padded-rectangle sampler,
  cyclic0.0–0.7, actual speaker selection, nonempty valid utterances and zero
  padding. At rate0.7 exactly one modality is observed.
- Keep: fixed COMMON evaluation masks from the completed preceding round,
  training seed/init/data order/RNG, features, split, environment and task loss.
  Test evaluation is RNG-isolated and must not affect the training sequence.
- Keep: original cfg84 no-JEPA Flat1600/Local Skip/CHM, and the identical
  one-stage three-candidate scalar Router. No new architecture or loss.
- Keep: batch32/Adam/wd1e-5/original dropout; backbone LR1e-3 for100epochs,
  then1e-4 for50. Router3e-4 then3e-5. Original scalar MSE.
- Keep runtime: biggpu, `/data2/yb/reproduction_workspace/envs/s0/bin/python`,
  Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2. GPU4 forbidden.

Preserve all previous validation-selected results and checkpoints read-only.
Those runs did not save every epoch's test predictions/weights, so their full
per-rate Test-best cannot be recovered merely by evaluating best.pt/last.pt.
New independent output directories are required; this is not a continuation.

The data manifest is verified byte-identically against accepted SHA
`e790eee2c850418fde9652ab85ca4a1eeb12b155c5651d91b6caea8b1e36c213`;
all6598actual feature/label file hashes remain checked. Protected old weights
serve only the exact code/data bridge, then are discarded before initialization.

## Artifacts and interpretation

Save eight `best_miss_0pX.pt`, matching same-evaluation predictions/masks/labels
and selected epochs, and complete last.pt with model/Adam/RNG/history/best refs.
Do not evaluate a selected checkpoint on a different randomly sampled mask and
attach those metrics to its original best score. Each rate's ACC accompanies
its W-F1-selected predictions. Final reporting uses those saved arrays.

An eight-rate mean may combine eight different epochs/checkpoints. It is an
internal oracle diagnostic, not one deployed model's independent test score.
Old80.747 used a different missing sampler; restoring selection alone does NOT
make its masks identical to these new results or guarantee recovering80points.

## Launch

From a frozen Git source snapshot:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_mosi_gcnet_testbest_20261020.run \
  --output /data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpus GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
         GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
         GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

Finite queue, one real Joint smoke, at most2training children/card, RAM/VRAM
admission, >=16GiB free disk for this scope, no killing peers or automatic retries.
End-of-queue report checks pairing, all rates, complete budgets and oracle policy.
