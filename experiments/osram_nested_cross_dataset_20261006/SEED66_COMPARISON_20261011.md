# Flat versus original Nested: seed66

INTERNAL DIAGNOSTIC ONLY

Eight-rate (0.0–0.7) equal-weight means, percentages. IEMOCAP uses equal
weight across five held-out sessions first. These are per-rate Test-oracle
BEST checkpoints, not final-epoch scores or three-seed means.

|Dataset|Epoch budget|Metric|Flat|Nested|Delta pp|
|---|---:|---|---:|---:|---:|
|MOSI|150|ACC|81.174|81.231|+0.057|
|MOSI|150|W-F1|81.068|81.163|+0.095|
|MOSEI|100|ACC|83.799|83.988|+0.189|
|MOSEI|100|W-F1|83.671|83.881|+0.210|
|IEMOCAP4|100|ACC|79.278|79.453|+0.175|
|IEMOCAP4|100|UA|79.404|79.647|+0.243|
|IEMOCAP6|100|ACC|61.378|60.879|-0.499|
|IEMOCAP6|100|UA|60.268|59.403|-0.865|

MOSI uses the paired completed seed66 continuation: original100 epochs
at1e-3 plus50 at1e-4, retaining BEST across all150. This is NOT the constant
LR150 experiment; that separate Nested run has W-F1=81.001067.
Other datasets use the original100epoch runs. Historical IEMOCAP Flat
selection_metric is unrecorded; Nested selects Accuracy, so this is a numerical
comparison, not a verified identical-selection ablation.

Sources (existing artifacts only; no new training or inference):
- `../osram_nested_low_lr_extend150_20261010/flat_FINAL_METRICS.json`
- `../osram_nested_low_lr_extend150_20261010/nested_FINAL_METRICS.json`
- `REQUESTED_METRICS_20261008.json`: datasets/model/seeds[seed=66]/mean8

MOSI values recomputed as100 times the arithmetic mean of eight `test`
accuracy/weighted_f1 values. Other rows read the audited seed66 aggregates
above, not the three-seed headline tables.
