# Hierarchical evidence gate — completed

Three seeds completed 100 epochs each on biggpu GPU0, all return codes zero.
Source e5e17f3; configurations and per-seed provenance/metrics archived in results.
Existing runner verifies canonical evaluation masks and eight best checkpoints
per seed before marking completion. No new inference or retraining for this summary.

**Per-rate Test-oracle internal diagnosis, not validation-selected paper results.**
W-F1 percentages, differences in percentage points. High missing = .5/.6/.7.

| Seed | Flat 8-rate | Gate 8-rate | Delta | Flat high | Gate high | Delta |
|---|---:|---:|---:|---:|---:|---:|
|66|81.068|80.097|−0.971|76.352|75.272|−1.080|
|67|80.556|80.056|−0.500|75.990|75.400|−0.589|
|68|80.053|79.901|−0.152|74.440|75.128|+0.688|
|Mean|80.559|80.018|−0.541|75.594|75.267|−0.327|

All three seeds decline on the 8-rate average. Only seed68 improves on high
missing rates. This run does not support an overall improvement from the
two-level gate. It does not isolate whether feature filtering or evidence
competition caused the decline. No additional ablation has been run here.

Raw summary: results/SUMMARY.json. Scores use each rate's saved best checkpoint,
not epoch100 scores. Baseline reused, not retrained; compare matched three seeds.
