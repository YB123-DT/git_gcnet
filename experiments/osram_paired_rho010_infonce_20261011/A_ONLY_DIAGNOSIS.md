# rho.10 task-only A-only polarity failure

INTERNAL DIAGNOSTIC ONLY. No training, parameter changes or threshold selection.

Frozen replay of completed C02 miss=.7 BEST (epoch100), original six-pattern
evaluator, original historical6e2a998 snapshot, biggpu GPU6. A-only replay W-F1
0.25403848060956324 exactly matches fixed_results.json. Captured original
_metrics inputs to count polarity on656 nonneutral labels, prediction>0.

| Quantity | Count/value |
|---|---:|
| Actual positive |277|
| Actual negative |379|
| Predicted positive |655|
| Predicted negative |1|
| True positive |277|
| True negative |1|
| False positive |378|
| False negative |0|
| Prediction mean,656 nonneutral samples |0.791387677192688|
| Accuracy |42.378048780487804%|
| Weighted F1 |25.403848060956324%|

This is nearly one-sided polarity prediction, NOT proof constant features or
constant Memory. Prediction std across all686 valid samples is0.251791507;
prediction/label correlation is0.136210214. Both signs still technically appear,
so predicted_sign_count=2 in fixed_results.json did not reveal this imbalance.

Prior rho.50 task-only A-only uses epoch43, std0.601731241, correlation0.151265838,
ACC59.908537%, W-F156.986495%. Correlation remains weak in both; much of the
classification difference is compatible with polarity calibration/distribution
rather than evidence of a proportionate semantic information gain. Causal
attribution is not established without paired distributions or controlled training.

Observed failure is selective: current Text-present scores remain about86%, AV
59.118% exceeds Flat57.958%, so the whole model is not uniformly collapsed.
Random miss=.7 selection yielded W-F1 76.556500%, but this does not select for
whole-conversation Audio-only quality: random and persistent history support
conditions differ. View2 independently deletes observed modalities per utterance;
it does not explicitly enforce six whole-conversation persistent patterns.

Possible explanations, NOT established causes: weaker augmentation supervision
at rho.10, distribution shift in missing-history support, and checkpoint-specific
polarity calibration. Epoch100 vs43 cannot alone establish overfitting; changing
rho does not guarantee monotonic robustness. This run has no InfoNCE contribution,
so do not attribute the failure to contrastive loss or a test-time projector.

No additional training, tuning or broad sweep recommended from this single-seed
observation. This note records a concrete output failure, not a discovered
architectural bottleneck or emotion-shift mechanism.
