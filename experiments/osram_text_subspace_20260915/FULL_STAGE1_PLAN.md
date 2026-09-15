# Seed66 Stage1 plus matched target-space audit

1. Run existing Stage1 exactly100 epochs; selected checkpoint remains minimum validation composite loss, not best Test or best sentiment. No loss changes, no new hyperparameters. Preserve all100 epoch train/validation metrics.
2. Load selected R, then instantiate a single fresh frozen-Teacher Student. Use one missing=.5 training batch, one model forward and one dropout realization. Both full-text256 and subspace32 losses consume the identical reg/cl predictions and Teacher targets.
3. Compare .1JEPA vs emotion and .05NCE vs .05reg on identical common-used encoder/OSRAM parameter coordinates. Report target-space norm ratios explicitly; no optimizer update in this comparison. A same-forward identity-projection test must recover identical results.
4. Summarize Stage1 trajectory and source-specific validation quality, plus whether auxiliary scale is comparable. Do not auto-sweep weights or temperature or launch Stage2 before interpreting this check.
