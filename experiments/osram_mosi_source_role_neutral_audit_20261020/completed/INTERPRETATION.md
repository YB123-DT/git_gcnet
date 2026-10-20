# What the three-seed12-slot diagnostic supports

INTERNAL DIAGNOSTIC ONLY

All24seed/rate BEST checkpoints and exact test masks replayed successfully.
No training,parameter update,new Memory/query trajectory or model change.
Tables are gamma restored to1,NOT source deletion(gamma0). Positive cells
mean the learned calibration helped relative to its single-slot neutralization.

## Main observation

|Seed|Original Flat W-F1|Same trained model,all gamma1|Learned weights|Learned minus neutral pp|
|---|---:|---:|---:|---:|
|66|79.128|79.407|79.450|+.043|
|67|78.751|78.615|78.578|-.036|
|68|78.794|78.471|78.471|+.001|

The final weight application changes very few polarity decisions. Its effect
on the eight-rate mean is small in each seed; the larger Flat differences
persist after all weights are restored to1. Thus the total between-model
score changes should NOT primarily be attributed to opposite12-slot learned
weight effects. Neutral-minus-Flat also includes different trained parameters
AND separately Test-best-selected epochs; it is not a pure causal estimate
of backbone training drift or evidence that the backbone necessarily learned
less useful information.

Three of12averaged cells have mixed positive/negative signs:Base/A,Base/T,
Gap-T/T. Some opposite-sign effects are only around.001pp. Remaining cells
are zero in one/more seeds or have consistent nonzero signs. The12independent
nonlinear intervention effects cannot be summed to explain overall W-F1.

## Actual strength of the learned calibration

|Seed|Active gamma min|max|mean|Neutral→learned corrections|harms|
|---|---:|---:|---:|---:|---:|
|66|.9532|.9824|.9696|7|5|
|67|.9753|.9831|.9795|4|6|
|68|.9888|1.0033|.9982|1|1|

Counts refer to5248nonzero-label evaluation exposures PER seed across eight
rates,not5248unique utterances. Each per-rate test has656nonzero labels and
686valid utterances. Gamma statistics pool valid active slots across selected
per-rate checkpoints/exposures; they are not semantic reliability scores.
All gamma values are near1,especially seed68. Zero table entries do not mean
the underlying source/read is absent or useless; polarity can remain unchanged
despite continuous prediction changes,or positive/negative effects can cancel.

## Reproducibility and limits

Execution_v2 completed in49.810s on biggpu GPU0,original s0 runtime. Exactly24
CHM scans (one test batch per condition); 360saved scalar prediction variants
were independently checked,maximum metric error0. Frozen state/RNG and exact
learned predictions/masks passed; cross-seed labels,IDs,masks and active slots
matched. Source reconstruction maximum error is recorded in FINAL_VERIFICATION.
All original checkpoints/predictions remain untouched,large audit NPZs remote.
The first attempt stopped BEFORE inference on tuple/list identity comparison;
source/output preserved,canonical JSON comparison repaired with regression.

This frozen intervention may be distribution-shifting. It does not prove
how a no-gate model would train,identify an optimal gate,or establish general
failure of source-aware designs. No labels trained a router or chose new
thresholds/weights. No extra modules,training or search was launched.
