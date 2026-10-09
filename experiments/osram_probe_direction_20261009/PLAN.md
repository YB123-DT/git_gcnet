# Frozen history-probe direction diagnostic

INTERNAL DIAGNOSTIC ONLY

Follow the user-specified design without fitting anything. Source completed audit `osram_frozen_memory_audit_20261009` (f2f5d0f execution, e359d95 archived results). Original cfg84 LARGE Flat seed66 per-rate checkpoints; existing Probe B target `preceding3_mean`. Test only, no validation; inherited Test-oracle selection disclosed. No new checkpoint/threshold/architecture search.

## Fixed scope and definitions

- All8rates. Primary family `recent_vs_earlier_T` only; retain BOTH original arms (nearest eligible Text deletion and next-earlier eligible Text deletion). Exactly reuse stored IDs/positions/modalities, no resampling or selecting by label/prediction. Count target/rate/arm pairs, not independent utterances.
- Primary probe initialization66, robustness67/68, reported separately and equally, never select best. Three probe seeds are not three independent backbones.
- R=[Base,Gap-A,Gap-T,Gap-V] forward512 each. D replicates the fitted train-only normalization, structural mask and fixed512->64 projection before its frozen515->64->1 MLP. Local and availability are constants. Gradient is with respect to RAW forward read values, not standardized coordinates.
- u=grad_R(D)/norm(grad_R(D)). h=dot(u,deltaR)u; o=deltaR-h. Compute projection in float64, cast perturbed reads to original model dtype for evaluation; report rounding/reconstruction/orthogonality errors. Gradient norm<=1e-12 uses h=0,o=deltaR and is counted, not silently dropped. Nonfinite active gradients fail.
- F is exactly original local_skip+emotion_adapter+emotion_norm+smax_fc. Keep current Local, mask, weights, inactive Gap and backward zero halves fixed. Compare baseline/full/hist/other reads. Reproduce cached original and historical-deletion predictions before interpretation.
- Cache complete original/deleted R and source IDs/labels/masks on biggpu, not Git. Hash source checkpoints/probe weights/preprocessing/deletion records and verify frozen states before/after. No optimizer or parameter updates.

## Analysis

Raw and signed changes Ds,Dy; past-three true-label probe MSE change and current-label original-model MSE change. Four large/small groups use per-rate/probe/arm medians of absolute changes, strictly descriptive, no coefficient selection. Also report Pearson/Spearman and partial correlations controlling log1p(norm deltaR), deletion lag, and current availability. No claim that these statistical controls prove causality.

Report component norms and norm fractions, task-head responses, corrections/harms, paired W-F1 and MSE, and nonlinear interaction `dy_full-dy_hist-dy_other`. Other denotes the local orthogonal complement, NOT proven non-emotional information; rank-one local tangent cannot capture all history information. Modified reads may lie off the memory trajectory manifold. Sensitivity is not semantic mediation.

## Implementation / verification

- `direction.py` + tests: exact probe recreation, gradient finite differences, masks, degenerate gradient and projection algebra; original readout replay.
- `run.py` + tests: exact stored-plan selection, deletion validity, independent memory scans, cached feature/probe/output parity, no source writes.
- `analyze.py` + tests: sample-aligned per-rate/arm/seed summaries and controls; conversation-level uncertainty, no pooled-rate pseudoreplication.
- One limited real-checkpoint check on healthy biggpuGPU2, then full8rate frozen replay onGPU2. GPU4 forbidden. Immutable source, persistent tmux, no duplicated launch; automatic summary. Commit/push code and completed aggregate results, never raw tensors/weights.
