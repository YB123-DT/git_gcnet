# Frozen observation-history drift diagnostic

No training, new loss, module, inference gate, or weight changes. Use original
cfg84 no-JEPA causal Flat seed66 per-rate best checkpoints, rates0–.7.
Validation/test reported separately. Historical checkpoint selection is
Test-oracle; all results INTERNAL diagnostics, not formal model performance.

For each baseline View1 use original split-specific masks(epoch0). Make four
View2 interventions: extra observed-slot deletion restricted to A,T,V,or all
modalities(mixed), probability.2, never adding observations, always retaining
at least one modality at a valid utterance. No deletion of raw feature files.
Use shared random deletion draws where possible for paired A/T/V comparison.

Anchor: current availability identical AND at least one strictly prior changed
availability in the same conversation. Compare actual current input and Local
for equality, with no future dependency. Eval/no_grad; independent zero Memory
per complete forward. Original code reads after decay and before current write.
Hooks observe original adapter inputs and returned post-Flat hidden/predictions;
they do not replay or alter the readout.

Numerical isolation: raw current inputs/masks must be bit-exact. The encoder
packs observed rows before projection, so history deletion changes GEMM shape.
A separate float32/float64 check found float32 Local absolute error 1.91e-6,
relative error 1.30e-7, versus exact zero in float64 for all four interventions.
The unchanged float32 diagnostic therefore requires Local absolute error <=1e-5
AND relative L2 <=1e-6; unaffected prefixes use atol1e-5/rtol1e-6.
All measured errors are retained, not rounded to zero. The initial overly strict
smoke failure is preserved; this does not authorize model/weight changes.

Record Local relative L2 change, Base/Gap cosine distance and norm differences,
post-Flat hidden drift, absolute prediction shift, sign/class changes. Only
effective first512context dimensions are measured; trailing512 must be zero.
Appending zeros does not dilute cosine mathematically. Zero-vector cosine is
undefined: exclude and report valid counts, never silently treat it as drift1.
Gap measurements only count currently missing modalities; retain fixedslots.

Metadata peranchor: seed/rate/split/conversationID/utteranceindex/currentpattern,
nominal intervention, actual modalities deleted in the strictprefix, cumulative
deleted bits, and distance to nearest strictly prior deletion. Distance bins
1,2,3–4,5+. These are multi-deletion trajectories, not single-event impulse
responses, so distance associations do not establish forgetting/decay rates.

Report each rate first, then unweighted rate macro, plus explicitly labeled
pooled counts/means. A/T/V intersection anchors provide a same-current-sample
comparison; deletion count/history eligibility may still differ across modes.
Mixed-prefix group labels distinguish single-type from multiple-type deletions.
Labels only support descriptive correct->wrong/wrong->correct counts, no fitting,
selection or threshold tuning. Follow original nonzero-label binary protocol
(prediction>0); mathematical sign changes may be reported separately.

Drift need not be harmful. Different history contains different real evidence.
Different layers' cosine distances are not directly an amplification ratio.
Single seed and inherited Test-oracle checkpoints limit generalization claims.

Server biggpu, hostGPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
GPU4 prohibited. Persist config/checkpoint/source hashes, full masks, per-anchor
measurements and frozen-state hashes. Never overwrite earlier diagnostics.
