# Correct IEMOCAP-6 checkpoint selection and training-fold scope

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

User authorization2026-10-20: rerun the mistaken validation-selected round;
delete its model checkpoints, retain its non-weight evidence, never touch the
protected original Flat baseline. This suite is finite: Flat/one-stage Router ×
seeds66/67/68 × five LOSO folds =30fresh100epoch runs. No new methods or HPO.

## Restored protocol

- Original cfg84 causal no-JEPA Flat1600, Local Skip, Block Delta joint write,
  write_step0.6,8heads,key/value64; original task CE (Router mixture NLL).
- Original GCNet random-mask-v1 host/guest padded-rectangle sampler, cyclic
  rates0.0–0.7; nonempty valid utterances, zero padding. Rate0.7 one observed
  modality. No persistent training, JEPA, completion, extra gate or new loss.
- Use all original training conversations from four sessions; test on held-out
  session. Remove the unauthorized10% adapter-training holdout. The old official
  loader aliases validation to the test session; this is explicitly Test-oracle,
  not an independent validation protocol. No validation criterion selects weights.
- Adam/backbone LR1e-3,Router LR3e-4,constant,wd1e-5,batch32,dropout0.5,
  original100epochs. Original data/features and cmerc-repro environment unchanged.
- Every epoch evaluate all eight rates. Each rate independently selects TEST ACC
  BEST, strict improvement, earliest ties. UA/W-F1/CE/recall must come from those
  exact same predictions. Keep all epoch scores, eightBEST and full last state.
- Fixed fold-specific legal GCNet evaluation banks, common across models and
  seeds, RNG-isolated. Preserve bank hashes and utterance/speaker/label identity.
  This is not the historical per-epoch resampled test mask protocol; do not claim
  identical trajectory or guaranteed recovery of61.714ACC.

Removing the holdout changes actual training data as well as selection. This is
restoration of the requested training protocol, NOT a selection-only ablation.
The erroneous scores remain archived and explicitly separate; no old validation
fold5 result is reused as Test-best. No old model weights initialize new runs.

## Artifacts

Each run:source/config/data/environment/seed/GPU provenance; full epoch history;
eight `best_miss_0pX.pt` with matching selected masks/labels/predictions/metrics;
`last.pt` with model/Adam/RNG/epoch and complete eight-rate best ledger. Save and
verify SHA256. Publishing copies selected arrays without random re-inference.
Suite summary requires30completed jobs and verifies actual paired train masks,
initializations, test cohorts and same-call metrics before reporting fold means.
Mean8 may combine eight different epochs; it is not one deployable checkpoint.

## Server and queue

biggpu only; Python `/data2/yb/conda_envs/cmerc-repro/bin/python`,
Torch1.13.1+cu117/NumPy1.26.4/sklearn1.3.2. Physical GPU4 forbidden.
Use three healthy UUID-bound GPUs, maximum two children per queue/card subject
to real smoke memory, freeVRAM/RAM/disk gates. No peer task is killed or moved.
An optional `--after-queue` can enforce an explicit dependency; otherwise run
independently on available resources. New immutable source/output paths only.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
  experiments.osram_iemocap6_gcnet_testbest_20261020.run \
  --output /data2/yb/remote_experiments/osram_iemocap6_gcnet_testbest_20261020/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpus GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
         GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
         GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

Cleanup plan/receipt records only the erroneous IEMOCAP weights. Original
MELD/MOSI weights, all exact mask files, predictions and accepted original
IEMOCAPTest-best weights are excluded. Cleanup does not make archived evidence
replayable after weights are deleted; this limitation is explicit.
