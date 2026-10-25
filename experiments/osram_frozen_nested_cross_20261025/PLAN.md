# Matched frozen Nested extension: three datasets, seed66

User approved MOSEI, IEMOCAP-6 and MELD after the completed MOSI frozen Nested
screen. Implement in this independent directory; preserve all previous runs.

## Locked experiment

- Exactly56formal jobs: MOSEIseed66eight own-rate parents, MELDseed66eight,
  IEMOCAPSixseed66five LOSO folds times8rates. No four-class or extra seeds/HPO.
- Same original Nested159235parameters via the already tested FrozenNestedModel
  in `osram_frozen_nested_gcnet_20261025/model.py`. Freeze/eval all parent modules;
  update only the existing Nested at the original Flat input. Raw LocalSkip,
  original CHM and joint BlockWrite/Query/causal order are unchanged.
- Own accepted Flat checkpoints and exact saved masks, feature files and splits
  reused from the completed cyclic Signature preparation. Runtime cmerc-repro.
  Parent learning rates are not stage2 LR: MOSEIparent1e-4, MELDparent1e-5,
  IEMOCAPparent1e-3. Do not retrain or redefine parents.
- Stage2 matches Signature budgets: MOSEI60epochs, MELD50, IEMOCAPSix100;
  constant3e-4, Adam WD1e-5,batch32, original cyclicGCNet-random-mask-v1.
- Original regression MSE including gold0 for MOSEI; original CE for MELD/IEMO.
  Earliest strict own-rate Test-best W-F1 for MOSEI and ACC for classifications.
  UA/WF1 come from the ACC-selected same predictions; preserve per-class recall.
- INTERNAL TEST-ORACLE SCREEN; means involve separate rate-specific frozen
  parents/blocks. IEMOCAP report equal-fold means and separately pooledOOF.

## Implementation and checks

1. New protocol/run.py: reuse current MOSI frozen adapter and trainer utilities,
   prepare datasets before importing process-global trainer/config, and select
   the existing pinned numerical source per dataset. Require parent config/
   source/environment/data/hash checks and exact selected-logit replay.
2. Tests-first finite dataset/seed/fold/rate/budget/selection cases; metric
   shapes, neutral treatment, one causal scan, frozen gradient/update and
   optimizer/RNG restore use the existing verified utilities.
3. New scheduler/summarizer: exactly56jobs; one real two-update GPU smoke per
   dataset, then finite capacity-gated queue on physicalGPU0/1/2 only,max4/card.
   Stagger starts and preserve unrelated tasks. Full block/Adam/RNG/history
   recovery, immutable source snapshots, hashes and persistent tmux controller.
4. Every run verifies actual cyclic samples/masks/update traces against the
   completed same-dataset/seed/fold/rate Signature reference; no added loss,
   mask draws in evaluation or test metric/selection substitution.
5. Summarize only completed validated artifacts; recompute selected predictions,
   report all rates/folds and branch-on/off corrections/harms against own Flat.
   Save/push scoped code, tests, launch and small result records. Preserve every
   protected checkpoint/cache/prediction and do not modify old source snapshots.

One bounded subagent owns protocol/runner/tests, leader owns queue/results/
deployment; a read-only review checks integration before formal admission.
