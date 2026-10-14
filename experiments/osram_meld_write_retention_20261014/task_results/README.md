# Completed fixed-original-Query task replay

INTERNAL DIAGNOSTIC ONLY. Full eight-rate replay completed without training.
Source is the retained, reproduced original large Flat seed66 (`b4a3138`),
Local Skip retained, canonical GCNet saved masks. No width512/no-Skip/legacy
weight substitution was made.

All eight original saved predictions and labels matched exactly. Normal branch
maximum absolute logit error was 2.384185791015625e-6; before/after model state
hashes matched at every rate; CUDA RNG remained unchanged and no gradient
was accumulated. Distance1 causal boundary checks passed during real replay.
Full inference wall time was 296.72 seconds; this is not training time.

Runner commit: `eb7e2e7`. Remote root:
`/data2/yb/remote_experiments/osram_meld_fixed_query_task_20261014`.
Raw per-pair compressed rows remain in `full/`; full class recalls, confusion
matrices, supports and direction counts remain in `analysis/TASK_AGGREGATE.json`.

Copied result artifacts retain original values:

- `SUMMARY.json`: provenance, all checkpoint hashes and all rate invariance checks.
- `TASK_AGGREGATE.csv`: paired task/state groups.
- `TASK_AGGREGATE.json.gz`: lossless compressed full aggregate, including recalls.
- `RESULT.md`: generated cohort table, NOT full-test selected performance.
- `SMOKE_SUMMARY.json`: separately labeled one-batch correctness record.

Targets are deduplicated across observed source modalities at each fixed
distance in the primary task table. Source-modal rows serve association/task
pairing. No combined cross-distance whole-test score is created. Positive
task ΔCE means harm; positive source reconstruction error change means weaker
association. Joint future writes are changed for all observed columns together,
not isolated Text-only writes. Recovery of an old Text association does not by
itself establish emotion-task benefit, Query failure or classifier waste.
