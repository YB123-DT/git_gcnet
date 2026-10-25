# Fixed-rate stage-two NeuMiss and Deep Signature, MOSI66

INTERNAL TEST-ORACLE SCREEN. Authorized finite experiment: two methods × eight
rates, with no new CSDI run. Existing one-stage experiments are not changed.

For each rate r independently:

1. Load the corresponding already-completed Flat seed66 per-rate BEST from
   `osram_mosi_cmerc_20261023/execution_v2`. Verify checkpoint SHA256, epoch,
   selected rate and original identity. Stage1 was the accepted cyclic Flat;
   it is not retrospectively described as fixed-rate training.
2. Freeze its entire encoder/CHM/Flat/LayerNorm/task head in eval mode.
3. Initialize that rate's own NeuMiss or Signature and train only the new branch.
   TRAIN rate is always r. There is NO cyclic selector and NO other training
   rate. Use the unchanged GCNet random-mask-v1 algorithm, host/guest behavior,
   padding and original sampler seeds within this specified rate.
4. Evaluate ONLY TEST rate r using its exact previously saved legal test mask;
   select strictly improved W-F1 at this same rate, earliest ties. Never evaluate
   the other seven rates for this run. Baseline predictions must reproduce the
   accepted parent before training.

Each branch is independent across rates. This is 16 logical conditions, not
16 seeds. NeuMiss is identically zero with all modalities observed, so rate0 is
a verified baseline replay with zero optimizer updates. Signature can operate
on complete-modality history, so its rate0 does train. Total:15 training jobs.

Stage2: seed66,150epochs, batch32, Adam WD1e-5, LR3e-4 first100 then3e-4/10 final50.
Original MOSI MSE includes gold0; original polarity ACC/W-F1 excludes gold0.
Original parent weights never update. Frozen weights remain differentiable
with respect to the new correction, so task gradients reach the branch.
No new reconstruction, relationship, consistency, or contrastive loss.

Tests cover per-rate parent selection, only-current-rate masks, no cyclic
schedule lookup, frozen-head input gradients and rate0 behavior. Before formal
jobs, one real full-batch two-update smoke at rate0.7 per method checks gradient
and checkpoint-replay paths. Smoke is separate and never reported as performance.

GPU0/3 are explicitly UUID bound, at most4 NEW task processes/GPU, admitted
against current free memory and measured smoke peak. Existing jobs are not
stopped, paused or moved; physical GPU4 is forbidden.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler \
  --root /data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1 \
  --source-commit ACTUAL_SOURCE_SHA \
  --gpu0-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
  --gpu3-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

Each run records effective fixed-rate config separately from original parent
config, SHA256 parent/mask/prediction provenance, selected correction weights,
exact predictions, history and complete correction/Adam/RNG `last.pt` recovery.
Frozen parents remain referenced protected files, not duplicated or overwritten.
The finite queue writes aggregate/per-rate Markdown, JSON and CSV after all
conditions finish and provenance/epoch checks pass.

Limits: two stages have extra optimization compared with the original parent;
do not attribute gains solely to architecture or call this equal total budget.
Per-rate original-parent Test-oracle plus stage2 Test-oracle creates selection
bias; report only internal exploration, not independent test generalization.
No automatic extra seeds, datasets, hyperparameter search or third method.
