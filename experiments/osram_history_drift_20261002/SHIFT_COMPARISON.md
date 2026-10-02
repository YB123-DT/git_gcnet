# Audit: proposed vector correction versus existing MemoryShiftFilter

Read-only code/result audit; no implementation or new inference/training.
Existing implementation: gcnet_missing_m3/osram.py MemoryShiftFilter (line400),
readout integration around1440. Actual old remote run:
`/data2/yb/remote_experiments/osram_cfg84_memory_shift_residual_20260928/runs`.
Remote status complete, smoke_only false. Seeds66/67/68 each complete100epochs
and each retain8best checkpoints. Local old README's pending status was stale.

|Component|Existing trained module|Newly proposed, not implemented|
|---|---|---|
|Anchor|Trainable original Flat pre-norm|Same|
|Local/memory relation space|Separate Local/shared memory projections,128d|Same|
|Evidence type|Four128d embeddings added to memory projection|Same|
|Relation features|q,k,k-q,abs(k-q),q*k,type|Same|
|Evidence correction|sigmoid(MLP(features))*(k-q)|MLP(features) outputs128d vector|
|Aggregation|Sum active corrections / number active|Same|
|Residual output|Zero-initialized Linear128→output_dim|Same|
|Placement|Add residual before original emotion_norm|Same|
|Memory/query/write|Unchanged|Same proposed|
|Loss/masks|Original single-view no-JEPA task objective|Same proposed|

For fixed learned q,k, the old evidence correction is collinear with k-q and
scaled by a scalar in(0,1). The proposal can generate a different128d direction
and magnitude. This is a genuine expressivity change, but not a new memory
mechanism; its benefit is untested. Projections already learn in the old model,
so scalar filtering must not be described as a completely fixed correction space.
The subtraction of separately learned projections is also not, by itself, a
verified semantic-conflict measure.

Existing three-seed Test-oracle mean W-F1:

|Model|Eight-rate|High .5/.6/.7|
|---|---:|---:|
|Flat|80.559|75.594|
|Old MemoryShiftFilter|79.949|75.206|
|Delta pp|−0.610|−0.387|

Eight-rate seed-matched deltas:66−.571,67−.160,68−1.098.
These are internal per-rate Test-oracle results, not formal validation-selected
paper results. The current suggestion should be described as a vector-valued
ablation of a previously unsuccessful residual route, not a fresh validated
module. There is no evidence yet that scalar direction constraints caused the
old degradation; greater flexibility alone is not a reason to expect higher W-F1.
