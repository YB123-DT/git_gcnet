# Completed three-seed comparison

INTERNAL TEST-ORACLE DIAGNOSTIC — NOT A FORMAL VALIDATION-SELECTED PAPER RESULT.

All seeds66/67/68 completed100 epochs on biggpu GPU0; child exit codes0.
All24 best per-rate checkpoints retained remotely. Each run passed the original
canonical evaluation-mask check. Summary means independently checked against
archived per-seed metrics. Original Flat reference was reused, not retrained.

| Metric (W-F1 %) | Flat | Evidence Gate | Paired delta (pp) |
|---|---:|---:|---:|
| Eight-rate mean | 80.559 ± 0.507 | 80.286 ± 0.195 | -0.273 ± 0.385 |
| High missing (.5/.6/.7) | 75.594 ± 1.016 | 75.955 ± 0.759 | +0.361 ± 0.372 |

Uncertainty is sample standard deviation across the three seeds, not standard
error. Delta dispersion is computed on paired per-seed differences.

| Seed | Flat eight-rate | Gate eight-rate | Delta | Flat high | Gate high | Delta |
|---|---:|---:|---:|---:|---:|---:|
|66|81.068|80.357|-0.711|76.352|76.292|-0.060|
|67|80.556|80.435|-0.121|75.990|76.488|+0.498|
|68|80.053|80.065|+0.012|74.440|75.086|+0.646|

The gate does not improve the overall mean. High-missing average improves
slightly in two seeds and declines slightly in one; this is not evidence of a
consistent or statistically established gain. Three seeds and Test-oracle
selection limit generalization. Do not replace the Flat baseline on this evidence.

Method: gamma0.2, lambda0.001 once-weighted active-valid gate regularization,
four scalar evidence gates, original trainable Flat and Memory, no-JEPA cyclic
random-missing protocol. No test-label oracle correction tables enter training.
Launch code af37f56; detailed configuration/provenance in results/ and LAUNCH.md.
No extra training, inference, or hyperparameter search was run for this summary.
