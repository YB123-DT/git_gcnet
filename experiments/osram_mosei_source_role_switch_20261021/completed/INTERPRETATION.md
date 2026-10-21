# MOSEI Source-Role: a useful within-checkpoint scale,not an overall model gain

INTERNAL DIAGNOSTIC ONLY. Frozen seed66/eight existing BEST checkpoints.
No retraining,weight edits,new masks,threshold selection or extra Memory scan.

## 1. The learned calibration does affect this checkpoint

| Scope | Learned gamma W-F1 | Gamma=1 W-F1 | Learned−neutral(pp) | Corrections | Harms |
|---|---:|---:|---:|---:|---:|
| Eight-rate mean |83.243611|82.883011|+0.360600|408|342|
| High missing |79.980188|79.471762|+0.508427|205|176|

Counts are repeated rate exposures,not unique independent utterances.
Learned output reproduces all eight saved predictions exactly. Every batch
has one causal Memory scan,shared by both readouts;frozen state and RNG match.

Thus the MOSI near-zero direct-contribution result must NOT be assumed for
MOSEI. Gamma is not near1 and neutralizing it does change some decisions.

## 2. It mostly amplifies existing reads,with very little source differentiation

All active gamma across rates range1.48008–1.85948;mean1.77895,
population variance0.00569430. No active gamma is within0.01of1 or below1.
Source means:A1.77917,T1.77871,V1.77897.

Pooled variance mixes different selected checkpoints and observation rows.
It does not prove source specificity. A stricter comparison uses gamma for
the SAME utterance/read-role across its active sources. Among58,962such rows
with at least two active sources,the mean source standard deviation is
0.000105540;maximum0.0140706. This supports weak average within-row source
differentiation,not exactly identical weights everywhere.

Compute this statistic from each saved rate NPZ as follows (no inference):

```python
counts = active.sum(-1)
row_mean = gamma.sum(-1) / np.maximum(counts, 1)
source_std = np.sqrt(
    np.where(active, (gamma - row_mean[..., None]) ** 2, 0).sum(-1)
    / np.maximum(counts, 1)
)
values = source_std[counts >= 2]
# Concatenate values across rates,then report mean/max;counts are exposures.
```

The learned behavior is therefore closer to a broadly similar positive
amplification than clear A/T/V selection. It is NOT proof that a single fixed
global coefficient would reproduce the same outputs;that intervention was
not requested and was not run.

## 3. The task effect depends on current/past Text availability

| Current/past Text condition | W-F1 gain(pp) | ACC gain(pp) | Corrections | Harms |
|---|---:|---:|---:|---:|
| T visible |−0.008119|−0.022692|157|157|
| T missing,pastT seen |+1.361842|+0.973416|223|146|
| T missing,pastT unseen |+0.682987|−0.182119|28|39|

These are equal nonempty-rate means within fixed non-label-defined groups.
The last group is NOT an unambiguous benefit:its mean W-F1 rises but ACC falls
and harms exceed corrections. Per-rate sample/class distributions and W-F1
weighting mean that macro-rate W-F1 gains need not have the sign of pooled
correction-minus-harm counts. Do not hide this metric disagreement.

The strongest positive descriptive signal is currentT missing with actual
pastT support. This does not prove the Router learned source-specific
cooperation or that Text-source evidence alone caused the gain.

## 4. Why this does not contradict the earlier lack of overall improvement

| Model/readout | Eight-rate W-F1 |
|---|---:|
| Separately trained original Flat |83.279076|
| Trained Direct checkpoint,gamma forced1 |82.883011|
| Same Direct checkpoint,learned gamma |83.243611|

The arithmetic decomposition is:

`−0.035465 = −0.396065 + 0.360600` percentage points.

The learned calibration is useful relative to neutralizing this NEW trained
checkpoint,yet the full model still fails to exceed the original Flat.
This is consistent with joint co-adaptation,but does not causally establish
that the original backbone was damaged or Flat absorbed the calibration.
The gamma1 readout uses new trained parameters,not the old Flat weights.

Conclusion:reject the claim that MOSEI calibration is numerically inactive.
Also reject the stronger claim that source-aware cooperation or an overall
architectural improvement has been established. Keep these two questions
separate. No new training or additional intervention is launched.
